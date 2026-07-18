import {
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  type CandidateCount,
  type FinalizationChallenge,
  type InputMode,
  neuroSelectApi,
  type ProfileSummary,
  type RankedCandidate,
  type SessionAction,
  type SessionView,
} from "./api";

const candidateCounts: CandidateCount[] = [4, 6, 8, 12];
const timingOptions = [900, 1400, 2200, 3500];

const reasonLabels: Record<string, string> = {
  missing_neural_evidence: "No neural evidence",
  low_neural_support: "Low neural support",
  low_neural_margin: "Neural result is ambiguous",
  neural_language_conflict: "Neural and language evidence conflict",
  lm_dominance_detected: "Language prior may be dominating",
  low_fused_score: "Combined support is low",
  low_fused_margin: "Top candidates are too close",
  sensitive_candidate: "Sensitive content requires extra confirmation",
};

const controlLabels: Record<string, { label: string; description: string }> = {
  other: {
    label: "Other",
    description: "Leave this candidate set and return to composing.",
  },
  back: {
    label: "Back",
    description: "Remove the last confirmed span.",
  },
  cancel: {
    label: "Cancel session",
    description: "End this session without finalizing a message.",
  },
};

function percentage(value: number | null) {
  return value === null ? "Unavailable" : `${Math.round(value * 100)}%`;
}

function confirmedText(view: SessionView | null) {
  return view?.session.confirmed_spans.map((span) => span.text).join(" ") ?? "";
}

function isTextEntryTarget(target: EventTarget | null) {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    (target instanceof HTMLElement && target.isContentEditable)
  );
}

interface EvidenceListProps {
  ranked: RankedCandidate;
}

function EvidenceList({ ranked }: EvidenceListProps) {
  const { breakdown } = ranked;
  const sources = [
    ["Neural", percentage(breakdown.neural)],
    ["Language", percentage(breakdown.generic_language)],
    [
      "Personal",
      breakdown.personal_lift === 0
        ? "No measured lift"
        : `${breakdown.personal_lift > 0 ? "+" : ""}${Math.round(
            breakdown.personal_lift * 100,
          )}% lift`,
    ],
    ["Retrieved", percentage(breakdown.retrieval)],
  ];

  return (
    <div className="evidence-stack">
      <dl className="evidence-list">
        {sources.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <p className="score-line">
        Combined score <strong>{breakdown.total_score.toFixed(3)}</strong>
      </p>
      {breakdown.dominance_flags.length > 0 && (
        <p className="evidence-warning">
          Check: {breakdown.dominance_flags.join(", ").replaceAll("-", " ")}
        </p>
      )}
      {ranked.retrieval_hits.length > 0 && (
        <div className="retrieval-notes">
          <h4>Retrieved record provenance</h4>
          {ranked.retrieval_hits.map((hit) => (
            <p key={hit.record.record_id}>
              <strong>{hit.record.kind.replaceAll("_", " ")}:</strong>{" "}
              {hit.record.content} <span>({hit.record.source})</span>
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

interface CandidateCardProps {
  ranked: RankedCandidate;
  rejected: boolean;
  selectionBlocked: boolean;
  busy: boolean;
  shortcut?: string;
  onSelect: (candidateId: string) => void;
  onReject: (candidateId: string) => void;
}

function CandidateCard({
  ranked,
  rejected,
  selectionBlocked,
  busy,
  shortcut,
  onSelect,
  onReject,
}: CandidateCardProps) {
  const { candidate } = ranked;
  const enhanced = ranked.confirmation_level === "enhanced";
  const descriptionId = `candidate-${candidate.candidate_id}-description`;

  return (
    <article className="candidate-card" data-risk={ranked.risk_level}>
      <button
        type="button"
        className="candidate-choice"
        disabled={busy || rejected || selectionBlocked}
        onClick={() => onSelect(candidate.candidate_id)}
        aria-describedby={descriptionId}
        aria-keyshortcuts={shortcut}
        data-scan-target="true"
      >
        <span className="candidate-order">Choice {ranked.rank}</span>
        <strong>{candidate.text}</strong>
        <span className="candidate-confidence">
          Neural support: {percentage(ranked.breakdown.neural)}
        </span>
      </button>
      <div id={descriptionId} className="candidate-meta">
        <div className="source-badges" aria-label="Candidate source labels">
          <span>Language</span>
          <span>
            {ranked.breakdown.neural === null ? "Neural unavailable" : "Neural"}
          </span>
          {ranked.breakdown.personal_lift !== 0 && <span>Personal</span>}
          {ranked.retrieval_hits.length > 0 && <span>Retrieved</span>}
          {ranked.risk_level !== "none" && (
            <span>Risk: {ranked.risk_level}</span>
          )}
          {enhanced && <span>Extra confirmation</span>}
          {rejected && <span>Rejected</span>}
        </div>
        <details>
          <summary>Evidence and provenance</summary>
          <EvidenceList ranked={ranked} />
        </details>
        <button
          type="button"
          className="text-button reject-button"
          disabled={busy || rejected}
          onClick={() => onReject(candidate.candidate_id)}
        >
          Reject this choice
        </button>
      </div>
    </article>
  );
}

export default function App() {
  const [profiles, setProfiles] = useState<ProfileSummary[]>([]);
  const [profileId, setProfileId] = useState("");
  const [inputMode, setInputMode] = useState<InputMode>("simulation");
  const [view, setView] = useState<SessionView | null>(null);
  const [challenge, setChallenge] = useState<FinalizationChallenge | null>(
    null,
  );
  const [manualText, setManualText] = useState("");
  const [candidateCount, setCandidateCount] = useState<CandidateCount>(8);
  const [maximumPhraseTokens, setMaximumPhraseTokens] = useState(4);
  const [simulatedTargetIndex, setSimulatedTargetIndex] = useState(0);
  const [scanEnabled, setScanEnabled] = useState(false);
  const [scanIntervalMs, setScanIntervalMs] = useState(1400);
  const [highContrast, setHighContrast] = useState(false);
  const [lowMotion, setLowMotion] = useState(false);
  const [highRiskAcknowledged, setHighRiskAcknowledged] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("Loading synthetic profiles…");
  const selectionDialogRef = useRef<HTMLHeadingElement>(null);
  const finalDialogRef = useRef<HTMLHeadingElement>(null);

  const selectedProfile = profiles.find(
    (profile) => profile.profile_id === profileId,
  );
  const message = confirmedText(view);
  const ranking = view?.ranking ?? null;
  const rankedCandidates = ranking?.ranked_candidates ?? [];
  const languageCandidates = rankedCandidates.filter(
    (item) => item.candidate.kind !== "control",
  );
  const controlCandidates = rankedCandidates.filter(
    (item) => item.candidate.kind === "control",
  );
  const pendingCandidate = rankedCandidates.find(
    (item) =>
      item.candidate.candidate_id === view?.pending_selection_candidate_id,
  );
  const terminal =
    view?.session.state === "finalized" || view?.session.state === "cancelled";
  const selectionBlocked =
    view?.input_mode === "simulation" && ranking?.disposition !== "display";

  const rootClassName = [
    "app",
    highContrast ? "high-contrast" : "",
    lowMotion ? "low-motion" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const targetOptions = useMemo(
    () => Array.from({ length: candidateCount - 3 }, (_, index) => index),
    [candidateCount],
  );

  async function loadProfiles() {
    setBusy(true);
    setError("");
    try {
      const loaded = await neuroSelectApi.listProfiles();
      setProfiles(loaded);
      setProfileId((current) => current || loaded[0]?.profile_id || "");
      setNotice("Synthetic profiles ready. No session has started.");
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Could not load profiles.",
      );
      setNotice("");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void loadProfiles();
  }, []);

  useEffect(() => {
    if (simulatedTargetIndex > candidateCount - 4) {
      setSimulatedTargetIndex(0);
    }
  }, [candidateCount, simulatedTargetIndex]);

  useEffect(() => {
    if (pendingCandidate) {
      selectionDialogRef.current?.focus();
    }
  }, [pendingCandidate]);

  useEffect(() => {
    if (challenge) {
      finalDialogRef.current?.focus();
    }
  }, [challenge]);

  useEffect(() => {
    if (
      !scanEnabled ||
      busy ||
      challenge ||
      pendingCandidate ||
      !view ||
      !["selecting", "candidates_ready"].includes(view.session.state)
    ) {
      return;
    }
    let index = -1;
    const timer = window.setInterval(() => {
      const targets = Array.from(
        document.querySelectorAll<HTMLButtonElement>(
          '[data-scan-target="true"]:not(:disabled)',
        ),
      );
      if (targets.length > 0) {
        index = (index + 1) % targets.length;
        targets[index]?.focus();
      }
    }, scanIntervalMs);
    return () => window.clearInterval(timer);
  }, [busy, challenge, pendingCandidate, scanEnabled, scanIntervalMs, view]);

  useEffect(() => {
    function handleShortcut(event: KeyboardEvent) {
      if (
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        isTextEntryTarget(event.target) ||
        challenge ||
        pendingCandidate ||
        busy
      ) {
        return;
      }
      const number = Number(event.key);
      if (Number.isInteger(number) && number >= 1 && number <= 9) {
        const candidate = languageCandidates[number - 1];
        if (candidate && !selectionBlocked) {
          event.preventDefault();
          void applyAction("select", candidate.candidate.candidate_id);
        }
      }
      if (
        event.key.toLowerCase() === "r" &&
        view?.input_mode === "simulation" &&
        ranking
      ) {
        event.preventDefault();
        void applyAction("repeat");
      }
    }
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  });

  useEffect(() => {
    function handleDialogEscape(event: KeyboardEvent) {
      if (event.key !== "Escape" || busy) return;
      if (pendingCandidate) {
        event.preventDefault();
        void resolveSelection(false);
      } else if (challenge) {
        event.preventDefault();
        void closeFinalization();
      }
    }
    window.addEventListener("keydown", handleDialogEscape);
    return () => window.removeEventListener("keydown", handleDialogEscape);
  });

  async function updateView(
    operation: () => Promise<SessionView>,
    successMessage: string,
  ) {
    setBusy(true);
    setError("");
    try {
      const updated = await operation();
      setView(updated);
      setNotice(successMessage);
      return updated;
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "The request failed.",
      );
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function startSession(event: FormEvent) {
    event.preventDefault();
    if (!profileId) return;
    const created = await updateView(
      () => neuroSelectApi.createSession(profileId, inputMode),
      "Session started. Confirmed text is empty.",
    );
    if (created) {
      setChallenge(null);
      setManualText("");
    }
  }

  async function startRound() {
    if (!view) return;
    await updateView(
      () =>
        neuroSelectApi.startRound(view.session.session_id, {
          simulatedTargetIndex,
          candidateCount,
          maximumPhraseTokens,
        }),
      "Candidate round ready. No candidate has been selected.",
    );
  }

  async function applyAction(action: SessionAction, candidateId?: string) {
    if (!view) return;
    const updated = await updateView(
      () =>
        neuroSelectApi.applyAction(
          view.session.session_id,
          action,
          candidateId,
        ),
      action === "select"
        ? "Selection recorded. Check whether another confirmation is required."
        : `${action.charAt(0).toUpperCase()}${action.slice(1)} action recorded.`,
    );
    if (updated?.session.state === "cancelled") {
      setNotice("Session cancelled. No final message was shared.");
    }
  }

  async function submitManualText(event: FormEvent) {
    event.preventDefault();
    if (!view || !manualText.trim()) return;
    const updated = await updateView(
      () =>
        neuroSelectApi.appendManualText(view.session.session_id, manualText),
      "Keyboard text explicitly added to the confirmed message.",
    );
    if (updated) setManualText("");
  }

  async function resolveSelection(accept: boolean) {
    if (!view || !pendingCandidate) return;
    await updateView(
      () =>
        neuroSelectApi.resolveSelection(
          view.session.session_id,
          pendingCandidate.candidate.candidate_id,
          accept,
        ),
      accept
        ? "Candidate confirmed and added to the message."
        : "Candidate rejected. It remains unavailable for this round.",
    );
  }

  async function requestFinalization() {
    if (!view) return;
    setBusy(true);
    setError("");
    try {
      const nextChallenge = await neuroSelectApi.requestFinalization(
        view.session.session_id,
      );
      setChallenge(nextChallenge);
      setHighRiskAcknowledged(false);
      setNotice("Review the exact final text before confirming.");
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not prepare confirmation.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function closeFinalization() {
    if (!view) return;
    const updated = await updateView(
      () => neuroSelectApi.rejectFinalization(view.session.session_id),
      "Finalization cancelled. Continue editing the message.",
    );
    if (updated) setChallenge(null);
  }

  async function confirmFinalization() {
    if (!view || !challenge) return;
    const updated = await updateView(
      () =>
        neuroSelectApi.confirmFinalization(
          view.session.session_id,
          challenge,
          highRiskAcknowledged,
        ),
      "Message finalized after explicit confirmation.",
    );
    if (updated) setChallenge(null);
  }

  function handleCandidateKeyDown(
    event: ReactKeyboardEvent,
    candidateId: string,
  ) {
    if (event.key === "Delete") {
      event.preventDefault();
      void applyAction("reject", candidateId);
    }
  }

  return (
    <div className={rootClassName}>
      <a
        className="skip-link"
        href="#workspace"
        inert={Boolean(pendingCandidate || challenge) || undefined}
      >
        Skip to communication workspace
      </a>
      <header
        className="site-header"
        inert={Boolean(pendingCandidate || challenge) || undefined}
      >
        <div>
          <p className="eyebrow">Local research interface</p>
          <h1>NeuroSelect</h1>
        </div>
        <p className="header-note">
          Suggestions stay provisional until you explicitly select and finalize
          them.
        </p>
      </header>

      <main
        id="workspace"
        className="workspace"
        tabIndex={-1}
        inert={Boolean(pendingCandidate || challenge) || undefined}
      >
        <aside className="boundary" aria-label="Research boundary">
          <strong>Research boundary:</strong> NeuroSelect reduces selection
          effort; it does not decode unrestricted thoughts. This step uses
          seeded simulation or keyboard input, not live EEG.
        </aside>

        <section className="status-region" aria-label="Application status">
          {notice && (
            <p role="status" aria-live="polite">
              {notice}
            </p>
          )}
          {error && (
            <p role="alert" className="error-message">
              <strong>Request failed:</strong> {error}{" "}
              {!view && (
                <button
                  type="button"
                  className="text-button"
                  onClick={() => void loadProfiles()}
                >
                  Retry profile loading
                </button>
              )}
            </p>
          )}
        </section>

        {!view ? (
          <section
            className="panel setup-panel"
            aria-labelledby="setup-heading"
          >
            <div className="section-heading">
              <p className="step-label">Session setup</p>
              <h2 id="setup-heading">
                Choose a synthetic profile and input source
              </h2>
            </div>
            <form onSubmit={(event) => void startSession(event)}>
              <label>
                Synthetic communication profile
                <select
                  value={profileId}
                  onChange={(event) => setProfileId(event.target.value)}
                  disabled={busy || profiles.length === 0}
                >
                  {profiles.map((profile) => (
                    <option key={profile.profile_id} value={profile.profile_id}>
                      {profile.display_name}
                    </option>
                  ))}
                </select>
              </label>
              {selectedProfile && (
                <p className="field-note">{selectedProfile.style_summary}</p>
              )}

              <fieldset>
                <legend>Input source</legend>
                <label className="radio-card">
                  <input
                    type="radio"
                    name="input-mode"
                    value="simulation"
                    checked={inputMode === "simulation"}
                    onChange={() => setInputMode("simulation")}
                  />
                  <span>
                    <strong>Seeded neural simulation</strong>
                    <small>
                      Reproducible candidate probabilities for the research
                      demo.
                    </small>
                  </span>
                </label>
                <label className="radio-card">
                  <input
                    type="radio"
                    name="input-mode"
                    value="manual"
                    checked={inputMode === "manual"}
                    onChange={() => setInputMode("manual")}
                  />
                  <span>
                    <strong>Keyboard debug mode</strong>
                    <small>
                      Exercises the full interface with neural evidence marked
                      absent.
                    </small>
                  </span>
                </label>
              </fieldset>

              <button
                className="primary-button"
                type="submit"
                disabled={busy || !profileId}
              >
                {busy ? "Starting…" : "Start local session"}
              </button>
            </form>
          </section>
        ) : (
          <>
            <section
              className="message-panel panel"
              aria-labelledby="confirmed-heading"
            >
              <div className="section-heading inline-heading">
                <div>
                  <p className="step-label">Confirmed text</p>
                  <h2 id="confirmed-heading">Message in progress</h2>
                </div>
                <span className="state-badge">
                  State: {view.session.state.replaceAll("_", " ")}
                </span>
              </div>
              <p
                className={
                  message ? "confirmed-message" : "confirmed-message empty"
                }
              >
                {message || "Nothing has been confirmed yet."}
              </p>
              <p className="session-caption">
                Profile:{" "}
                {selectedProfile?.display_name ?? view.session.profile_id} ·
                Input:{" "}
                {view.input_mode === "simulation"
                  ? "seeded simulation"
                  : "keyboard debug"}
              </p>
            </section>

            {!terminal && (
              <section className="composer-layout">
                <div className="primary-column">
                  {view.session.state === "draft" && (
                    <section
                      className="panel action-panel"
                      aria-labelledby="next-action-heading"
                    >
                      <div className="section-heading">
                        <p className="step-label">Next action</p>
                        <h2 id="next-action-heading">Build the message</h2>
                      </div>
                      {view.input_mode === "manual" && (
                        <form
                          className="manual-form"
                          onSubmit={(event) => void submitManualText(event)}
                        >
                          <label htmlFor="manual-text">
                            Explicit keyboard text
                          </label>
                          <div>
                            <input
                              id="manual-text"
                              value={manualText}
                              onChange={(event) =>
                                setManualText(event.target.value)
                              }
                              maxLength={160}
                              placeholder="Type a confirmed word or short phrase"
                            />
                            <button
                              type="submit"
                              disabled={busy || !manualText.trim()}
                            >
                              Add text
                            </button>
                          </div>
                        </form>
                      )}
                      <div className="action-row">
                        <button
                          className="primary-button"
                          type="button"
                          onClick={() => void startRound()}
                          disabled={busy}
                        >
                          Generate candidate round
                        </button>
                        <button
                          type="button"
                          onClick={() => void requestFinalization()}
                          disabled={busy || !message}
                        >
                          Review final message
                        </button>
                      </div>
                    </section>
                  )}

                  {ranking && view.active_generation && (
                    <section
                      className="candidate-section"
                      aria-labelledby="candidates-heading"
                    >
                      <div className="section-heading candidate-heading">
                        <div>
                          <p className="step-label">
                            Explicit selection required
                          </p>
                          <h2 id="candidates-heading">
                            Ranked candidate choices
                          </h2>
                        </div>
                        <p>
                          {languageCandidates.length} language choices ·{" "}
                          {controlCandidates.length} controls
                        </p>
                      </div>

                      <div
                        className={`disposition disposition-${ranking.disposition}`}
                      >
                        <strong>
                          {ranking.disposition === "display" &&
                            "Candidates are ready."}
                          {ranking.disposition === "request_repeat" &&
                            "Repeat recommended."}
                          {ranking.disposition === "abstain" &&
                            "The ranker abstained."}
                        </strong>{" "}
                        {ranking.disposition === "display"
                          ? "Choose a candidate; nothing is appended automatically."
                          : view.input_mode === "manual"
                            ? "Neural evidence is unavailable in keyboard mode; you may still make an explicit debug selection."
                            : "Language choices are blocked until neural support improves or you leave this round."}
                        {ranking.reason_codes.length > 0 && (
                          <ul>
                            {ranking.reason_codes.map((reason) => (
                              <li key={reason}>
                                {reasonLabels[reason] ?? reason}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>

                      <p className="keyboard-hint">
                        Keyboard: Tab then Enter selects any enabled choice.
                        Keys 1–9 select language choices in displayed order;
                        Delete rejects a focused choice; R repeats simulated
                        evidence.
                      </p>
                      <div className="candidate-grid">
                        {languageCandidates.map((ranked, index) => (
                          <div
                            key={ranked.candidate.candidate_id}
                            onKeyDown={(event) =>
                              handleCandidateKeyDown(
                                event,
                                ranked.candidate.candidate_id,
                              )
                            }
                          >
                            <CandidateCard
                              ranked={ranked}
                              rejected={view.rejected_candidate_ids.includes(
                                ranked.candidate.candidate_id,
                              )}
                              selectionBlocked={selectionBlocked}
                              busy={busy}
                              shortcut={
                                index < 9 ? String(index + 1) : undefined
                              }
                              onSelect={(candidateId) =>
                                void applyAction("select", candidateId)
                              }
                              onReject={(candidateId) =>
                                void applyAction("reject", candidateId)
                              }
                            />
                          </div>
                        ))}
                      </div>

                      <h3 className="controls-heading">Round controls</h3>
                      <div className="control-grid">
                        {controlCandidates.map((ranked) => {
                          const action =
                            view.active_generation?.control_actions[
                              ranked.candidate.candidate_id
                            ];
                          const copy = controlLabels[action ?? "other"];
                          return (
                            <button
                              key={ranked.candidate.candidate_id}
                              type="button"
                              className="control-card"
                              onClick={() =>
                                void applyAction(
                                  "select",
                                  ranked.candidate.candidate_id,
                                )
                              }
                              disabled={busy}
                              data-scan-target="true"
                            >
                              <strong>{copy.label}</strong>
                              <span>{copy.description}</span>
                            </button>
                          );
                        })}
                      </div>
                    </section>
                  )}
                </div>

                <aside
                  className="secondary-column"
                  aria-label="Session controls and settings"
                >
                  <section
                    className="panel control-panel"
                    aria-labelledby="session-controls-heading"
                  >
                    <h2 id="session-controls-heading">Session controls</h2>
                    <div className="stacked-buttons">
                      {view.input_mode === "simulation" && ranking && (
                        <button
                          type="button"
                          onClick={() => void applyAction("repeat")}
                          disabled={busy}
                          aria-keyshortcuts="R"
                        >
                          Repeat neural selection
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => void applyAction("back")}
                        disabled={busy}
                      >
                        Back one span
                      </button>
                      <button
                        type="button"
                        onClick={() => void applyAction("clear")}
                        disabled={busy || !message}
                      >
                        Clear confirmed text
                      </button>
                      <button
                        className="danger-button"
                        type="button"
                        onClick={() => void applyAction("cancel")}
                        disabled={busy}
                      >
                        Cancel session
                      </button>
                    </div>
                  </section>

                  <section
                    className="panel settings-panel"
                    aria-labelledby="settings-heading"
                  >
                    <h2 id="settings-heading">Access and round settings</h2>
                    <label>
                      Candidate targets per round
                      <select
                        value={candidateCount}
                        onChange={(event) =>
                          setCandidateCount(
                            Number(event.target.value) as CandidateCount,
                          )
                        }
                      >
                        {candidateCounts.map((count) => (
                          <option key={count} value={count}>
                            {count} total ({count - 3} language + 3 controls)
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Maximum words per phrase
                      <select
                        value={maximumPhraseTokens}
                        onChange={(event) =>
                          setMaximumPhraseTokens(Number(event.target.value))
                        }
                      >
                        {[1, 2, 3, 4, 6, 8].map((count) => (
                          <option key={count} value={count}>
                            {count}
                          </option>
                        ))}
                      </select>
                    </label>
                    {view.input_mode === "simulation" && (
                      <label>
                        Simulated intended position
                        <select
                          value={simulatedTargetIndex}
                          onChange={(event) =>
                            setSimulatedTargetIndex(Number(event.target.value))
                          }
                        >
                          {targetOptions.map((index) => (
                            <option key={index} value={index}>
                              Position {index + 1}
                            </option>
                          ))}
                        </select>
                        <small>
                          Explicit research ground truth; never shown as
                          inferred text.
                        </small>
                      </label>
                    )}
                    <label className="toggle-row">
                      <input
                        type="checkbox"
                        checked={scanEnabled}
                        onChange={(event) =>
                          setScanEnabled(event.target.checked)
                        }
                      />
                      Enable automatic focus scanning
                    </label>
                    <label>
                      Focus scan timing
                      <select
                        value={scanIntervalMs}
                        onChange={(event) =>
                          setScanIntervalMs(Number(event.target.value))
                        }
                        disabled={!scanEnabled}
                      >
                        {timingOptions.map((milliseconds) => (
                          <option key={milliseconds} value={milliseconds}>
                            {(milliseconds / 1000).toFixed(1)} seconds
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="toggle-row">
                      <input
                        type="checkbox"
                        checked={highContrast}
                        onChange={(event) =>
                          setHighContrast(event.target.checked)
                        }
                      />
                      High-contrast mode
                    </label>
                    <label className="toggle-row">
                      <input
                        type="checkbox"
                        checked={lowMotion}
                        onChange={(event) => setLowMotion(event.target.checked)}
                      />
                      Low-motion mode
                    </label>
                  </section>

                  <section
                    className="panel metrics-panel"
                    aria-labelledby="metrics-heading"
                  >
                    <h2 id="metrics-heading">Session metrics</h2>
                    <dl>
                      <div>
                        <dt>Rounds</dt>
                        <dd>{view.metrics.round_count}</dd>
                      </div>
                      <div>
                        <dt>Selections</dt>
                        <dd>{view.metrics.selection_count}</dd>
                      </div>
                      <div>
                        <dt>Rejections</dt>
                        <dd>{view.metrics.rejection_count}</dd>
                      </div>
                      <div>
                        <dt>Repeats</dt>
                        <dd>{view.metrics.repeat_count}</dd>
                      </div>
                      <div>
                        <dt>Backtracks</dt>
                        <dd>{view.metrics.backtrack_count}</dd>
                      </div>
                      <div>
                        <dt>Manual additions</dt>
                        <dd>{view.metrics.manual_text_count}</dd>
                      </div>
                    </dl>
                  </section>

                  <section
                    className="panel replay-panel"
                    aria-labelledby="replay-heading"
                  >
                    <h2 id="replay-heading">EEG replay</h2>
                    <p>
                      Recorded P300 replay controls are intentionally
                      unavailable until the real dataset and stream adapter are
                      implemented.
                    </p>
                    <button type="button" disabled>
                      Load recording
                    </button>
                  </section>
                </aside>
              </section>
            )}

            {terminal && (
              <section className="panel terminal-panel" aria-live="polite">
                <p className="step-label">Session complete</p>
                <h2>
                  {view.session.state === "finalized"
                    ? "Message finalized"
                    : "Session cancelled"}
                </h2>
                <p>
                  {view.session.state === "finalized"
                    ? `Explicitly confirmed text: “${message}”`
                    : "No message was finalized or shared."}
                </p>
              </section>
            )}
          </>
        )}
      </main>

      {pendingCandidate && (
        <div className="dialog-backdrop">
          <section
            className="dialog-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="selection-dialog-heading"
            aria-describedby="selection-dialog-description"
          >
            <p className="step-label">Additional confirmation</p>
            <h2
              id="selection-dialog-heading"
              ref={selectionDialogRef}
              tabIndex={-1}
            >
              Confirm this candidate
            </h2>
            <p id="selection-dialog-description">
              You chose <strong>“{pendingCandidate.candidate.text}”</strong>. It
              was not the unqualified top choice or it requires stronger
              confirmation. It has not been added yet.
            </p>
            <EvidenceList ranked={pendingCandidate} />
            <div className="dialog-actions">
              <button
                className="primary-button"
                type="button"
                onClick={() => void resolveSelection(true)}
                disabled={busy}
              >
                Confirm and add candidate
              </button>
              <button
                type="button"
                onClick={() => void resolveSelection(false)}
                disabled={busy}
              >
                Reject candidate
              </button>
            </div>
          </section>
        </div>
      )}

      {challenge && (
        <div className="dialog-backdrop">
          <section
            className="dialog-panel final-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="final-dialog-heading"
            aria-describedby="final-dialog-description"
          >
            <p className="step-label">Required final confirmation</p>
            <h2 id="final-dialog-heading" ref={finalDialogRef} tabIndex={-1}>
              Confirm the exact final message
            </h2>
            <p id="final-dialog-description">
              This is the only text that will be finalized. Review every word.
            </p>
            <blockquote>{challenge.text}</blockquote>
            <p className="challenge-meta">
              Confirmation expires{" "}
              {new Date(challenge.expires_at).toLocaleTimeString()} · Text
              fingerprint {challenge.text_sha256.slice(0, 12)}…
            </p>
            {challenge.high_risk_acknowledgement_required && (
              <label className="risk-acknowledgement">
                <input
                  type="checkbox"
                  checked={highRiskAcknowledged}
                  onChange={(event) =>
                    setHighRiskAcknowledged(event.target.checked)
                  }
                />
                I reviewed the sensitive content and explicitly approve this
                exact message.
              </label>
            )}
            <div className="dialog-actions">
              <button
                className="primary-button"
                type="button"
                onClick={() => void confirmFinalization()}
                disabled={
                  busy ||
                  (challenge.high_risk_acknowledgement_required &&
                    !highRiskAcknowledged)
                }
              >
                Confirm and finalize exact text
              </button>
              <button
                type="button"
                onClick={() => void closeFinalization()}
                disabled={busy}
              >
                Continue editing
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
