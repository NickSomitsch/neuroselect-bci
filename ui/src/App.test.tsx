import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type {
  FinalizationChallenge,
  RankedCandidate,
  SessionView,
} from "./api";

const profile = {
  profile_id: "synthetic-concise",
  display_name: "Mara Vale",
  style_summary: "Direct, concise requests with short acknowledgments.",
  synthetic: true,
} as const;

function rankedCandidate(
  id: string,
  text: string,
  rank: number,
  options: {
    control?: boolean;
    neural?: number | null;
    enhanced?: boolean;
    retrieved?: boolean;
  } = {},
): RankedCandidate {
  const neural = options.neural === undefined ? 0.72 : options.neural;
  return {
    rank,
    candidate: {
      candidate_id: id,
      text,
      kind: options.control
        ? "control"
        : text.includes(" ")
          ? "phrase"
          : "word",
      origins: options.control ? ["application-control"] : ["generic-language"],
      risk_tags: [],
      retrieval_record_ids: options.retrieved ? ["record-water"] : [],
    },
    breakdown: {
      neural,
      generic_language: options.control ? 0 : rank === 1 ? 0.6 : 0.4,
      personal_lift: options.retrieved ? 0.08 : 0,
      retrieval: options.retrieved ? 0.7 : 0,
      diversity_adjustment: 0,
      risk: 0,
      weighted_contributions: { neural: neural === null ? 0 : neural * 0.65 },
      total_score: options.control ? 0.02 : rank === 1 ? 0.61 : 0.43,
      dominance_flags: [],
    },
    risk_level: "none",
    confirmation_level: options.enhanced ? "enhanced" : "standard",
    retrieval_hits: options.retrieved
      ? [
          {
            record: {
              record_id: "record-water",
              kind: "preference",
              content: "Mara prefers still water.",
              source: "synthetic:profile:synthetic-concise",
            },
            score: 0.7,
            matched_terms: ["water"],
            explanation: "Matched water.",
          },
        ]
      : [],
  };
}

function makeView(
  options: {
    mode?: "manual" | "simulation";
    state?: SessionView["session"]["state"];
    text?: string;
    round?: boolean;
    disposition?: "display" | "request_repeat" | "abstain";
    pending?: string | null;
  } = {},
): SessionView {
  const mode = options.mode ?? "manual";
  const round = options.round ?? false;
  const disposition =
    options.disposition ?? (mode === "manual" ? "abstain" : "display");
  const neural = mode === "manual" ? null : 0.72;
  const ranked = [
    rankedCandidate("candidate-hello", "Hello", 1, { neural, retrieved: true }),
    rankedCandidate("candidate-help", "I need help", 2, {
      neural: mode === "manual" ? null : 0.2,
      enhanced: true,
    }),
    rankedCandidate("control-other", "Other", 3, { control: true, neural }),
    rankedCandidate("control-back", "Back", 4, { control: true, neural }),
    rankedCandidate("control-cancel", "Cancel", 5, { control: true, neural }),
  ];
  return {
    session: {
      session_id: "session-ui-test",
      profile_id: profile.profile_id,
      state: options.state ?? (round ? "selecting" : "draft"),
      confirmed_spans: options.text
        ? [{ text: options.text, action_id: "action-one" }]
        : [],
      active_candidate_set_id: round ? "candidate-set-test" : null,
      provisional_candidate_id: options.pending ?? null,
      created_at: "2026-07-18T12:00:00Z",
      updated_at: "2026-07-18T12:00:00Z",
    },
    input_mode: mode,
    active_generation: round
      ? {
          candidate_set: {
            candidate_set_id: "candidate-set-test",
            context_sha256: "a".repeat(64),
            candidates: ranked.map((item) => item.candidate),
            generator_revision: "fixture-v1",
            prompt_revision: "prompt-v1",
          },
          generic_language_support: {
            "candidate-hello": 0.6,
            "candidate-help": 0.4,
          },
          control_actions: {
            "control-other": "other",
            "control-back": "back",
            "control-cancel": "cancel",
          },
          backend: {
            backend_id: "fixture",
            model_id: "fixture",
            model_revision: "1",
            generator_revision: "fixture-v1",
            prompt_revision: "prompt-v1",
            deterministic: true,
          },
          risk_policy_revision: "conservative-sensitive-content-v1",
          diagnostics: {
            raw_proposal_count: 10,
            selected_language_count: 2,
            unused_valid_count: 8,
            rejected_by_reason: {},
          },
        }
      : null,
    ranking: round
      ? {
          candidate_set_id: "candidate-set-test",
          policy_revision: "ranking-v1",
          neural_evidence_id: `${mode}-evidence-one`,
          disposition,
          confirmation_level: "standard",
          reason_codes:
            disposition === "display" ? [] : ["missing_neural_evidence"],
          ranked_candidates: ranked,
          fused_top_candidate_id: "candidate-hello",
          neural_top_candidate_id: mode === "manual" ? null : "candidate-hello",
          display_top_candidate_id:
            disposition === "display" ? "candidate-hello" : null,
          fused_margin: 0.18,
          neural_margin: mode === "manual" ? null : 0.52,
          requires_explicit_selection: true,
          automatic_selection_permitted: false,
        }
      : null,
    rejected_candidate_ids: [],
    pending_selection_candidate_id: options.pending ?? null,
    finalization_pending: options.state === "awaiting_final_confirmation",
    high_risk_acknowledgement_required: false,
    metrics: {
      round_count: round ? 1 : 0,
      selection_count: options.text ? 1 : 0,
      rejection_count: 0,
      repeat_count: 0,
      backtrack_count: 0,
      clear_count: 0,
      other_count: 0,
      manual_text_count: options.text ? 1 : 0,
    },
  };
}

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const fetchMock = vi.fn<typeof fetch>();

async function start(mode: "manual" | "simulation" = "manual") {
  fetchMock.mockResolvedValueOnce(jsonResponse([profile]));
  fetchMock.mockResolvedValueOnce(jsonResponse(makeView({ mode }), 201));
  render(<App />);
  await screen.findByRole("option", { name: "Mara Vale" });
  if (mode === "manual") {
    fireEvent.click(screen.getByRole("radio", { name: /Keyboard debug mode/ }));
  }
  fireEvent.click(screen.getByRole("button", { name: "Start local session" }));
  await screen.findByRole("heading", { name: "Message in progress" });
}

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("states the research boundary and exposes a fully keyboard-operable setup", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse([profile]));
    render(<App />);

    expect(
      screen.getByRole("heading", { level: 1, name: "NeuroSelect" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Research boundary")).toHaveTextContent(
      "does not decode unrestricted thoughts",
    );
    expect(
      await screen.findByLabelText("Synthetic communication profile"),
    ).toHaveValue(profile.profile_id);
    expect(
      screen.getByRole("radio", { name: /Seeded neural simulation/ }),
    ).toBeChecked();
    expect(
      screen.getByRole("button", { name: "Start local session" }),
    ).toBeEnabled();
  });

  it("shows separated provenance and only appends an explicitly selected candidate", async () => {
    await start("manual");
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        makeView({ mode: "manual", round: true, disposition: "abstain" }),
      ),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Generate candidate round" }),
    );

    expect(
      await screen.findByText("The ranker abstained."),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Neural unavailable").length).toBeGreaterThan(0);
    expect(
      screen.getByText("Retrieved", { selector: ".source-badges span" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Nothing has been confirmed yet."),
    ).toBeInTheDocument();

    fetchMock.mockResolvedValueOnce(
      jsonResponse(makeView({ mode: "manual", text: "Hello" })),
    );
    fireEvent.click(screen.getByRole("button", { name: /Choice 1 Hello/ }));

    await waitFor(() =>
      expect(
        screen.getByText("Hello", { selector: ".confirmed-message" }),
      ).toBeInTheDocument(),
    );
    const selectionRequest = JSON.parse(
      String(fetchMock.mock.calls.at(-1)?.[1]?.body),
    ) as Record<string, unknown>;
    expect(selectionRequest).toEqual({
      action: "select",
      candidate_id: "candidate-hello",
      explicit: true,
    });
  });

  it("requires enhanced candidate confirmation and exact-text final confirmation", async () => {
    await start("simulation");
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        makeView({ mode: "simulation", round: true, disposition: "display" }),
      ),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Generate candidate round" }),
    );
    await screen.findByRole("button", { name: /Choice 2 I need help/ });

    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        makeView({
          mode: "simulation",
          state: "awaiting_selection_confirmation",
          round: true,
          pending: "candidate-help",
        }),
      ),
    );
    fireEvent.click(
      screen.getByRole("button", { name: /Choice 2 I need help/ }),
    );
    expect(
      await screen.findByRole("dialog", { name: "Confirm this candidate" }),
    ).toHaveTextContent("It has not been added yet");
    expect(screen.getByRole("main")).toHaveAttribute("inert");

    fetchMock.mockResolvedValueOnce(
      jsonResponse(makeView({ mode: "simulation", text: "I need help" })),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Confirm and add candidate" }),
    );
    await screen.findByText("I need help", { selector: ".confirmed-message" });

    const challenge: FinalizationChallenge = {
      session_id: "session-ui-test",
      text: "I need help",
      text_sha256: "b".repeat(64),
      confirmation_nonce: "confirmation-nonce-123456",
      high_risk_acknowledgement_required: false,
      expires_at: "2026-07-18T12:05:00Z",
    };
    fetchMock.mockResolvedValueOnce(jsonResponse(challenge));
    fireEvent.click(
      screen.getByRole("button", { name: "Review final message" }),
    );
    expect(
      await screen.findByRole("dialog", {
        name: "Confirm the exact final message",
      }),
    ).toHaveTextContent("I need help");

    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        makeView({
          mode: "simulation",
          state: "finalized",
          text: "I need help",
        }),
      ),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Confirm and finalize exact text" }),
    );
    expect(
      await screen.findByRole("heading", { name: "Message finalized" }),
    ).toBeInTheDocument();

    const finalRequest = JSON.parse(
      String(fetchMock.mock.calls.at(-1)?.[1]?.body),
    ) as Record<string, unknown>;
    expect(finalRequest).toMatchObject({
      session_id: "session-ui-test",
      text_sha256: "b".repeat(64),
      confirmation_nonce: "confirmation-nonce-123456",
      explicit_confirmation: true,
    });
  });

  it("applies candidate-count, timing, contrast, and keyboard selection settings", async () => {
    await start("simulation");
    fireEvent.change(screen.getByLabelText("Candidate targets per round"), {
      target: { value: "6" },
    });
    fireEvent.change(screen.getByLabelText("Maximum words per phrase"), {
      target: { value: "2" },
    });
    fireEvent.click(screen.getByLabelText("Enable automatic focus scanning"));
    expect(screen.getByLabelText("Focus scan timing")).toBeEnabled();
    fireEvent.change(screen.getByLabelText("Focus scan timing"), {
      target: { value: "2200" },
    });
    fireEvent.click(screen.getByLabelText("High-contrast mode"));
    fireEvent.click(screen.getByLabelText("Low-motion mode"));
    expect(
      screen.getByLabelText("High-contrast mode").closest(".app"),
    ).toHaveClass("high-contrast", "low-motion");

    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        makeView({ mode: "simulation", round: true, disposition: "display" }),
      ),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Generate candidate round" }),
    );
    await screen.findByRole("button", { name: /Choice 1 Hello/ });

    const roundRequest = JSON.parse(
      String(fetchMock.mock.calls.at(-1)?.[1]?.body),
    ) as Record<string, unknown>;
    expect(roundRequest).toMatchObject({
      candidate_count: 6,
      maximum_phrase_tokens: 2,
    });

    fetchMock.mockResolvedValueOnce(
      jsonResponse(makeView({ mode: "simulation", text: "Hello" })),
    );
    fireEvent.keyDown(window, { key: "1" });
    await screen.findByText("Hello", { selector: ".confirmed-message" });
    expect(
      JSON.parse(String(fetchMock.mock.calls.at(-1)?.[1]?.body)),
    ).toMatchObject({
      action: "select",
      candidate_id: "candidate-hello",
    });
  });

  it("blocks language selection on simulated abstention while leaving repeat available", async () => {
    await start("simulation");
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        makeView({
          mode: "simulation",
          round: true,
          disposition: "request_repeat",
        }),
      ),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Generate candidate round" }),
    );

    expect(await screen.findByText("Repeat recommended.")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Choice 1 Hello/ }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Repeat neural selection" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Cancel session", hidden: false }),
    ).toBeEnabled();
  });
});
