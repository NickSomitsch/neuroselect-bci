export type CandidateCount = 4 | 6 | 8 | 12;
export type InputMode = "manual" | "simulation";
export type SessionState =
  | "draft"
  | "generating"
  | "candidates_ready"
  | "selecting"
  | "awaiting_selection_confirmation"
  | "awaiting_final_confirmation"
  | "finalized"
  | "cancelled";

export type SessionAction =
  "select" | "reject" | "repeat" | "back" | "clear" | "cancel" | "other";

export interface ProfileSummary {
  profile_id: string;
  display_name: string;
  style_summary: string;
  synthetic: true;
}

export interface Candidate {
  candidate_id: string;
  text: string;
  kind: "word" | "phrase" | "control" | "character";
  origins: string[];
  risk_tags: string[];
  retrieval_record_ids: string[];
}

export interface RetrievalHit {
  record: {
    record_id: string;
    kind: string;
    content: string;
    source: string;
  };
  score: number;
  matched_terms: string[];
  explanation: string;
}

export interface RankedCandidate {
  rank: number;
  candidate: Candidate;
  breakdown: {
    neural: number | null;
    generic_language: number;
    personal_lift: number;
    retrieval: number;
    diversity_adjustment: number;
    risk: number;
    weighted_contributions: Record<string, number>;
    total_score: number;
    dominance_flags: string[];
  };
  risk_level: "none" | "elevated" | "high";
  confirmation_level: "standard" | "enhanced";
  retrieval_hits: RetrievalHit[];
}

export interface SessionView {
  session: {
    session_id: string;
    profile_id: string;
    state: SessionState;
    confirmed_spans: Array<{ text: string; action_id: string }>;
    active_candidate_set_id: string | null;
    provisional_candidate_id: string | null;
    created_at: string;
    updated_at: string;
  };
  input_mode: InputMode;
  active_generation: {
    candidate_set: {
      candidate_set_id: string;
      context_sha256: string;
      candidates: Candidate[];
      generator_revision: string;
      prompt_revision: string;
    };
    generic_language_support: Record<string, number>;
    control_actions: Record<string, "other" | "back" | "cancel">;
    backend: {
      backend_id: string;
      model_id: string;
      model_revision: string;
      generator_revision: string;
      prompt_revision: string;
      deterministic: boolean;
    };
    risk_policy_revision: string;
    diagnostics: {
      raw_proposal_count: number;
      selected_language_count: number;
      unused_valid_count: number;
      rejected_by_reason: Record<string, number>;
    };
  } | null;
  ranking: {
    candidate_set_id: string;
    policy_revision: string;
    neural_evidence_id: string;
    disposition: "display" | "request_repeat" | "abstain";
    confirmation_level: "standard" | "enhanced";
    reason_codes: string[];
    ranked_candidates: RankedCandidate[];
    fused_top_candidate_id: string;
    neural_top_candidate_id: string | null;
    display_top_candidate_id: string | null;
    fused_margin: number;
    neural_margin: number | null;
    requires_explicit_selection: true;
    automatic_selection_permitted: false;
  } | null;
  rejected_candidate_ids: string[];
  pending_selection_candidate_id: string | null;
  finalization_pending: boolean;
  high_risk_acknowledgement_required: boolean;
  metrics: {
    round_count: number;
    selection_count: number;
    rejection_count: number;
    repeat_count: number;
    backtrack_count: number;
    clear_count: number;
    other_count: number;
    manual_text_count: number;
  };
}

export interface FinalizationChallenge {
  session_id: string;
  text: string;
  text_sha256: string;
  confirmation_nonce: string;
  high_risk_acknowledgement_required: boolean;
  expires_at: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      } else if (payload.detail) {
        detail = JSON.stringify(payload.detail);
      }
    } catch {
      // Keep the status text when the response is not JSON.
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

function sessionPath(sessionId: string, suffix = "") {
  return `/api/v1/sessions/${encodeURIComponent(sessionId)}${suffix}`;
}

export const neuroSelectApi = {
  listProfiles: () => request<ProfileSummary[]>("/api/v1/profiles"),

  createSession: (profileId: string, inputMode: InputMode) =>
    request<SessionView>("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId, input_mode: inputMode }),
    }),

  startRound: (
    sessionId: string,
    options: {
      simulatedTargetIndex: number;
      candidateCount: CandidateCount;
      maximumPhraseTokens: number;
    },
  ) =>
    request<SessionView>(sessionPath(sessionId, "/rounds"), {
      method: "POST",
      body: JSON.stringify({
        simulated_target_index: options.simulatedTargetIndex,
        candidate_count: options.candidateCount,
        maximum_phrase_tokens: options.maximumPhraseTokens,
      }),
    }),

  applyAction: (
    sessionId: string,
    action: SessionAction,
    candidateId?: string,
  ) =>
    request<SessionView>(sessionPath(sessionId, "/actions"), {
      method: "POST",
      body: JSON.stringify({
        action,
        ...(candidateId ? { candidate_id: candidateId } : {}),
        explicit: true,
      }),
    }),

  resolveSelection: (sessionId: string, candidateId: string, accept: boolean) =>
    request<SessionView>(sessionPath(sessionId, "/selection-confirmation"), {
      method: "POST",
      body: JSON.stringify({
        candidate_id: candidateId,
        accept,
        explicit_confirmation: true,
      }),
    }),

  appendManualText: (sessionId: string, text: string) =>
    request<SessionView>(sessionPath(sessionId, "/manual-text"), {
      method: "POST",
      body: JSON.stringify({ text, explicit_confirmation: true }),
    }),

  requestFinalization: (sessionId: string) =>
    request<FinalizationChallenge>(sessionPath(sessionId, "/finalization"), {
      method: "POST",
    }),

  confirmFinalization: (
    sessionId: string,
    challenge: FinalizationChallenge,
    highRiskAcknowledged: boolean,
  ) =>
    request<SessionView>(sessionPath(sessionId, "/finalization/confirm"), {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        text_sha256: challenge.text_sha256,
        confirmation_nonce: challenge.confirmation_nonce,
        explicit_confirmation: true,
        high_risk_acknowledged: highRiskAcknowledged,
      }),
    }),

  rejectFinalization: (sessionId: string) =>
    request<SessionView>(sessionPath(sessionId, "/finalization/reject"), {
      method: "POST",
    }),
};
