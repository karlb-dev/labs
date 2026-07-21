const LABS = [
  [1,"Residual Stream & Logit Lens","Trace how a prediction emerges over depth, beginning with a microscope smoke test that verifies the instrument before the science.","intro","lab01_residual_logit_lens",["OBS","DECODE"],["residual stream","logit lens","instrumentation"]],
  [2,"Direct Logit Attribution","Measure which attention and MLP components push toward—or away from—the model’s answer.","intro","lab02_direct_logit_attribution",["ATTR"],["DLA","components","decomposition"]],
  [3,"Attention Routing","Find head motifs and induction behavior, then test whether the apparent routing actually matters.","intro","lab03_attention_routing",["ATTR","CAUSAL"],["attention","heads","induction"]],
  [4,"Probing with Controls","Ask what is linearly decodable, whether it is selective, and how a truth direction transfers across statement families.","intro","lab04_probing_controls",["DECODE"],["probes","truth","controls"]],
  [5,"Activation Patching","Localize where a fact is causally recovered with clean/corrupt pairs, patching maps, and specificity controls.","intro","lab05_patching_causal_tracing",["CAUSAL"],["patching","causal tracing","factual recall"]],
  [6,"Manual Circuit Discovery","Build a small subgraph and test its faithfulness, completeness, minimality, and held-out transfer.","intro","lab06_circuit_discovery",["ATTR","CAUSAL"],["circuits","ablation","faithfulness"]],
  [7,"Steering & Refusal","Construct dose-response curves, monitor a refusal direction, and examine representation engineering’s dual-use boundary.","intro","lab07_steering_refusal",["DECODE","CAUSAL"],["steering","refusal","representation engineering"]],
  [8,"SAEs & Transcoders","Move from superposition to feature dictionaries: find, label, validate, and causally clamp learned features.","intro","lab08_sae_transcoders",["DECODE","CAUSAL"],["SAE","transcoders","superposition"]],
  [9,"Attribution Graphs","Build a transcoder replacement model, trace feature-level circuits, and validate graph-guided interventions on the real model.","intro","lab09_attribution_graphs",["ATTR","CAUSAL"],["graphs","transcoders","circuit tracing"]],
  [10,"Chain-of-Thought Faithfulness","Stress-test reasoning traces with hint injection, necessity curves, inserted mistakes, and matched filler controls.","intro","lab10_cot_faithfulness",["OBS","CAUSAL"],["reasoning","CoT","faithfulness"]],
  [11,"Mechanistic Reliability Audit","Reconcile the claim ledger against fresh evidence and deliver a fixed-schema audit, safety case, and rebuttal.","intro","lab11_reliability_audit",["AUDIT"],["capstone","reliability","claim ledger"]],
  [12,"Relation Geometry","Re-run the introductory toolkit across 12 controlled relation families and audit the operationalization itself.","advanced","lab12_relation_geometry",["DECODE","CAUSAL"],["geometry","relations","method validation"]],
  [13,"Emotion Geometry","Compare read and write directions for affect, test transfer, and separate emotion from sentiment and arousal confounds.","advanced","lab13_emotion_geometry",["DECODE","CAUSAL"],["emotion","steering","confounds"]],
  [14,"Certainty & Calibration","Relate internal answerability, entropy, margin, hedging style, and generated verbal confidence.","advanced","lab14_certainty_calibration",["DECODE"],["certainty","calibration","hedging"]],
  [15,"Multi-Turn Instrumentation","Validate chat-template spans, cache parity, boundary patching, and null traces before making longitudinal claims.","advanced","lab15_multiturn_harness",["AUDIT"],["multi-turn","cache parity","instrumentation"]],
  [16,"Sycophancy & User Belief","Separate truth, user-belief, agreement, politeness, and certainty directions under misconception pressure.","advanced","lab16_sycophancy_user_belief",["DECODE","CAUSAL"],["sycophancy","belief","agreement"]],
  [17,"Persona, Voice & Register","Extract paired style directions, test held-out transfer, steer neutral prompts, and trace traits across turns.","advanced","lab17_persona_voice_register",["DECODE","CAUSAL"],["persona","roleplay","register"]],
  [18,"Humor as Incongruity","Measure setup entropy and ending surprisal, inspect attention routing, and ask whether steering makes output genuinely funnier.","advanced","lab18_humor_incongruity",["ATTR","CAUSAL"],["humor","surprisal","attention"]],
  [19,"Model Diffing with Crosscoders","Map shared, base-only, and instruct-only features between related models, with taxonomy and intervention controls.","advanced","lab19_model_diffing_crosscoders",["DECODE","CAUSAL"],["crosscoders","model diffing","feature atlas"]],
  [20,"Benign Model Organisms","Construct sealed, auditable model organisms with leak scans, commitments, baseline tests, and spillover checks.","advanced","lab20_model_organisms",["CONSTRUCTION","AUDIT"],["model organisms","blind audit","safety"]],
  [21,"LoRA & Safety Depth","Locate where adapter training lives, test wrappers, and compare base-versus-instruct safety depth.","advanced","lab21_lora_safety_depth",["ATTR","AUDIT"],["LoRA","localization","safety"]],
  [22,"Evaluation Awareness","Learn eval-versus-natural directions, test cross-format transfer, and audit safe steering and mention behavior.","advanced","lab22_eval_awareness",["DECODE","CAUSAL"],["eval awareness","steering","controls"]],
  [23,"Blind Audit","Preregister, inspect a public package, submit claims before unsealing, and score them against hidden behavior.","advanced","lab23_blind_audit",["AUDIT"],["preregistration","blind audit","scoring"]],
  [24,"Knowledge Conflict","Track context-versus-parametric competition, revision under pressure, and the quadrants where behavior and internals disagree.","advanced","lab24_belief_revision",["DECODE","CAUSAL"],["belief revision","knowledge conflict","multi-turn"]],
  [25,"Find the Wire","Inject concept states and test whether self-reports track the intervention, its source, and the false-positive floor.","advanced","lab25_find_the_wire",["SELF-REPORT","CAUSAL"],["self-report","injection","grounding"]],
  [26,"Causal Abstraction","Write formal hypotheses and test them with behavior-preserving residual resampling plus breaking and wrong-site donors.","special","lab26_causal_abstraction",["FORMAL","CAUSAL"],["resampling","causal abstraction","formal hypotheses"]],
  [27,"Path-Specific Mediation","Distinguish node effects from source-to-receiver path proxies with mediated, joint, reverse, and random patches.","special","lab27_path_mediation",["CAUSAL"],["mediation","paths","patching"]],
  [28,"Editing & Unlearning","Apply reversible localized activation edits and audit paraphrase transfer, retain sets, neighbors, and side effects.","special","lab28_editing_unlearning",["CAUSAL","AUDIT"],["editing","unlearning","side effects"]],
  [29,"Circuit Birth","Watch behavior, probes, attention motifs, and intervention transfer emerge across a controlled checkpoint sequence.","special","lab29_training_dynamics",["OBS","CAUSAL"],["training dynamics","checkpoints","feature lineage"]],
  [30,"Cross-Layer Feature Lineage","Track supervised prototype directions across depth with split, merge, confusable, and transfer screens.","special","lab30_feature_lineage",["DECODE"],["feature geometry","lineage","cross-layer"]],
  [31,"Automated Interpretability","Audit generated feature labels at scale with held-out tests, calibration, abstention frontiers, and human review queues.","special","lab31_auto_interp",["AUDIT"],["auto-interp","feature labels","calibration"]],
  [32,"Reward & Preference Circuits","Study a DPO-style preference proxy while testing length, politeness, agreement, sentiment, and refusal shortcuts.","special","lab32_reward_preference",["ATTR","CAUSAL"],["reward models","preference","DPO"]],
  [33,"Multimodal Interpretability","Run a synthetic connector audit with image, text, caption, OCR, alignment, leak, and patch controls.","special","lab33_multimodal_mechanistic",["AUDIT","CAUSAL"],["multimodal","connector","alignment"]],
  [34,"Tool Use & State Tracking","Probe controlled tool choice and state at prompt boundaries, then test interventions and corrupted-result reliance.","special","lab34_tool_use_state",["DECODE","CAUSAL"],["agents","tool use","state tracking"]],
  [35,"Reproducible Paper Capstone","Bind claims to frozen runs, survive adversarial review, repair failures, and ship a reproducible evidence package.","special","lab35_reproducible_capstone",["AUDIT"],["capstone","reproducibility","paper"]],
  [36,"Severance Report Channel","Verify report-channel hypotheses with B2 screening, B3 bridging, B4 matched-output source attribution, and B5 insertion detection.","research","lab36_severance_report_channel",["SELF-REPORT","CAUSAL"],["Severance","report channel","source attribution"]]
].map(([id,title,description,phase,slug,evidence,tags]) => ({id,title,description,phase,slug,evidence,tags}));

const phaseNames = {intro:"Intro sequence",advanced:"Advanced course",special:"Special topics",research:"Research extension"};
const phaseResultNames = {all:"complete course",intro:"Intro · 01—11",advanced:"Advanced · 12—25",special:"Special topics · 26—35",research:"Research · 36"};
const repo = "https://github.com/karlb-dev/labs/blob/main/interpretability/labs/";
let activePhase = "all";

const grid = document.querySelector("#lab-grid");
const search = document.querySelector("#lab-search");
const empty = document.querySelector("#empty-state");
const count = document.querySelector("#result-count");
const resultPhase = document.querySelector("#result-phase");

function renderLabs() {
  const needle = search.value.trim().toLowerCase();
  const shown = LABS.filter(lab => {
    const phaseMatch = activePhase === "all" || lab.phase === activePhase;
    const haystack = `${lab.id} ${lab.title} ${lab.description} ${lab.evidence.join(" ")} ${lab.tags.join(" ")}`.toLowerCase();
    return phaseMatch && (!needle || haystack.includes(needle));
  });

  grid.innerHTML = shown.map(lab => `
    <a class="lab-card ${lab.phase}" href="${repo}${lab.slug}.md" target="_blank" rel="noreferrer">
      <div class="lab-top"><span class="lab-number">${String(lab.id).padStart(2,"0")}</span><small>${phaseNames[lab.phase]}</small><i>↗</i></div>
      <h3>${lab.title}</h3>
      <p>${lab.description}</p>
      <div class="evidence">${lab.evidence.map(tag => `<span>${tag}</span>`).join("")}</div>
    </a>`).join("");
  count.textContent = `${String(shown.length).padStart(2,"0")} labs`;
  resultPhase.textContent = phaseResultNames[activePhase];
  empty.hidden = shown.length !== 0;
}

document.querySelectorAll("[data-phase]").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-phase]").forEach(item => item.classList.remove("active"));
    button.classList.add("active");
    activePhase = button.dataset.phase;
    renderLabs();
  });
});

search.addEventListener("input", renderLabs);
document.addEventListener("keydown", event => {
  if (event.key === "/" && document.activeElement !== search) {
    event.preventDefault();
    search.focus();
  }
  if (event.key === "Escape" && document.activeElement === search) {
    search.value = "";
    search.blur();
    renderLabs();
  }
});

const heat = [0.08,0.12,0.18,0.12,0.08,0.1,0.2,0.36,0.44,0.2,0.12,0.32,0.72,0.88,0.38,0.16,0.38,0.82,1,0.52,0.1,0.24,0.48,0.62,0.28,0.06,0.12,0.22,0.3,0.14];
document.querySelector("#heatmap").insertAdjacentHTML("afterbegin", heat.map(value => `<i style="--heat:${value}"></i>`).join(""));
renderLabs();
