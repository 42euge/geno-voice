const variants = [
  { key: "A", name: "Capsule · expands in place" },
  { key: "B", name: "Command palette · question first" },
  { key: "C", name: "Side glance · stays out of the way" },
];

const root = document.querySelector("#prototype");
const switcher = document.querySelector("#switcher");
const label = document.querySelector("#variant-label");
let state = "answer";
let timer;

const wave = () => `
  <div class="waveform" aria-label="Listening">
    ${[11, 25, 18, 29, 15, 23, 31, 18, 26, 13, 22, 16]
      .map((height) => `<i style="--peak:${height}px"></i>`).join("")}
  </div>`;

const dots = () => `<div class="dots" aria-label="Thinking"><i></i><i></i><i></i></div>`;

const answerCopy = `HTTP 429 means the service is rate-limiting you: too many requests arrived in a short window. Wait for the <code>Retry-After</code> interval, then retry with exponential backoff and jitter. If it keeps happening, reduce concurrency or request a higher quota.`;

function variantA() {
  return `<section class="variant-a state-${state}" data-surface>
    <div class="surface glass">
      <div class="capsule-row">
        <span class="brand-mark">✦</span>
        <div class="capsule-copy" data-listening><strong>Listening…</strong><span>Ask a quick question</span></div>
        <div class="capsule-copy" data-thinking><strong>Looking into it</strong><span>What does HTTP 429 mean?</span></div>
        <div class="capsule-copy" data-answer><strong>Quick answer</strong><span>Geno Agent</span></div>
        <div data-listening>${wave()}</div>
        <div data-thinking>${dots()}</div>
        <span class="hint"><kbd>⌥</kbd><kbd>Space</kbd></span>
      </div>
      <div class="a-answer" data-answer>
        <div class="divider"></div>
        <p class="eyebrow">You asked</p>
        <div class="question">“What does an HTTP 429 response mean, and what should I do?”</div>
        <p class="answer">${answerCopy}</p>
        <div class="answer-actions"><button>Copy answer</button><button>Ask another</button><span class="spacer"></span><span class="hint"><kbd>esc</kbd> close</span></div>
      </div>
    </div>
  </section>`;
}

function variantB() {
  return `<section class="variant-b state-${state}" data-surface>
    <div class="surface glass">
      <div class="palette-input">
        <span class="brand-mark">✦</span>
        <span class="spoken ${state === "listening" ? "muted" : ""}">${state === "listening" ? "Ask anything…" : "What does an HTTP 429 response mean?"}</span>
        <div data-listening>${wave()}</div>
        <div data-thinking>${dots()}</div>
        <span class="hint"><kbd>⌥ Space</kbd></span>
      </div>
      <div class="palette-body" data-answer>
        <p class="eyebrow">Concise answer</p>
        <p class="answer">${answerCopy}</p>
        <div class="palette-footer"><span>↵ Ask a follow-up</span><span>⌘ C Copy</span><span>esc Close</span></div>
      </div>
    </div>
  </section>`;
}

function variantC() {
  return `<section class="variant-c state-${state}" data-surface>
    <div class="surface glass">
      <header class="side-header">
        <span class="brand-mark">✦</span>
        <div><strong>Geno Quick Agent</strong><span>${state === "answer" ? "Answered just now" : "One-off question"}</span></div>
        <span class="side-close">×</span>
      </header>
      <div class="side-listening" data-listening>${wave()}<p>Go ahead, I’m listening</p></div>
      <div class="side-listening" data-thinking>${dots()}<p>Finding the short version…</p></div>
      <div data-answer>
        <div class="side-question">What does an HTTP 429 response mean, and what should I do?</div>
        <div class="side-answer"><p class="eyebrow">The short version</p><p class="answer">${answerCopy}</p></div>
      </div>
    </div>
  </section>`;
}

function currentIndex() {
  const key = new URLSearchParams(location.search).get("variant")?.toUpperCase();
  const index = variants.findIndex((variant) => variant.key === key);
  return index < 0 ? 0 : index;
}

function render() {
  const variant = variants[currentIndex()];
  label.textContent = `${variant.key} — ${variant.name}`;
  root.innerHTML = variant.key === "A" ? variantA() : variant.key === "B" ? variantB() : variantC();
  root.querySelector("[data-surface]").addEventListener("click", replay);
}

function choose(index) {
  const wrapped = (index + variants.length) % variants.length;
  const url = new URL(location.href);
  url.searchParams.set("variant", variants[wrapped].key);
  history.replaceState({}, "", url);
  state = "answer";
  clearTimeout(timer);
  render();
}

function replay() {
  clearTimeout(timer);
  state = state === "answer" ? "listening" : state === "listening" ? "thinking" : "answer";
  render();
  if (state === "listening") timer = setTimeout(() => { state = "thinking"; render(); timer = setTimeout(() => { state = "answer"; render(); }, 900); }, 1200);
}

document.querySelector("#previous").addEventListener("click", () => choose(currentIndex() - 1));
document.querySelector("#next").addEventListener("click", () => choose(currentIndex() + 1));
window.addEventListener("popstate", render);
window.addEventListener("keydown", (event) => {
  if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName) || document.activeElement?.isContentEditable) return;
  if (event.key === "ArrowLeft") choose(currentIndex() - 1);
  if (event.key === "ArrowRight") choose(currentIndex() + 1);
  if (event.code === "Space") replay();
});

// The prototype switcher never belongs in a deployed surface.
const isLocalPrototype = location.protocol === "file:" || ["localhost", "127.0.0.1"].includes(location.hostname);
switcher.hidden = !isLocalPrototype;
choose(currentIndex());
