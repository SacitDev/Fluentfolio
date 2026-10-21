// Fluentfolio's own script. The page security policy runs scripts from files only, so every handler is
// attached here, never written into the page. Its jobs: the upload, the copy buttons, the audio player's
// keys and time buttons, the live progress of an analysis, and the busy mark of a button whose request runs.
"use strict";

// Copy the text of a read-only box. If the browser refuses, select the text so the user can copy it.
function copyText(button) {
  const id = button.dataset.copy;
  const box = document.getElementById(id);
  const result = document.querySelector('[data-copy-result="' + id + '"]');
  const say = (text) => {
    if (result) result.textContent = text;
  };
  const selectInstead = () => {
    box.focus();
    box.select();
    say("The text is selected. Press Ctrl+C to copy it.");
  };
  if (!navigator.clipboard || !window.isSecureContext) {
    selectInstead();
    return;
  }
  const count = box.value.length.toLocaleString("en-US");
  navigator.clipboard.writeText(box.value).then(() => say("Copied " + count + " characters."), selectInstead);
}

// Play the recording from a time button's time, in ms; a preview button also says how long to play.
let stopAt = null; // where a preview ends, in seconds
function seek(button) {
  const player = document.getElementById("player");
  if (!player) return;
  const from = Number(button.dataset.seek) / 1000;
  player.currentTime = from;
  stopAt = button.dataset.length ? from + Number(button.dataset.length) / 1000 : null;
  player.play().catch(() => {}); // a browser may refuse to start sound; the time is still set
}

// A preview pauses at its end. Media events do not bubble, so this listens in the capture phase; a learner
// who moves the player far past the end, or pauses it, has taken over and the preview is forgotten.
document.addEventListener("timeupdate", (event) => {
  if (stopAt === null || event.target.id !== "player") return;
  const at = event.target.currentTime;
  if (at >= stopAt && at < stopAt + 1) event.target.pause();
  if (at >= stopAt) stopAt = null;
}, true);
document.addEventListener("pause", () => { stopAt = null; }, true);

document.addEventListener("click", (event) => {
  const copy = event.target.closest("button[data-copy]");
  if (copy) copyText(copy);
  const time = event.target.closest("button[data-seek]");
  if (time) seek(time);
});

// Send a chosen recording as the raw request body, which the server saves as it arrives. The answer (what
// the file holds, or an error card) goes into the element that the input's data-upload names.
function sendFile(input) {
  const file = input.files[0];
  if (!file) return;
  const target = document.getElementById(input.dataset.upload);
  const bar = document.querySelector('progress[data-upload-progress="' + input.id + '"]');
  const finish = () => {
    input.removeAttribute("aria-busy");
    if (bar) bar.hidden = true;
  };
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/upload?slot=" + encodeURIComponent(input.dataset.slot || "upload"));
  xhr.setRequestHeader("HX-Request", "true");
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
  if (bar) {
    bar.value = 0;
    bar.hidden = false;
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) bar.value = event.loaded / event.total;
    });
  }
  xhr.addEventListener("load", () => {
    htmx.swap(target, xhr.responseText, { swapStyle: "innerHTML" });
    finish();
  });
  xhr.addEventListener("error", () => {
    target.textContent = "The upload stopped before the end. Choose the file again.";
    finish();
  });
  input.setAttribute("aria-busy", "true");
  xhr.send(file);
}

document.addEventListener("change", (event) => {
  const input = event.target;
  if (input.matches("input[type=file][data-upload]")) sendFile(input);
  // The token field's Show box: the field shows its text while the box is ticked.
  if (input.matches("input[data-show]")) {
    document.getElementById(input.dataset.show).type = input.checked ? "text" : "password";
  }
});

// While a request runs, the button that sent it is marked busy for screen readers; the page greys it out
// meanwhile (hx-disabled-elt) and shows a status line under it.
function markBusy(event, busy) {
  const source = event.detail.elt;
  const button = source.matches("form") ? source.querySelector("button[type=submit]") : source;
  if (!button || !button.matches("button")) return;
  if (busy) button.setAttribute("aria-busy", "true");
  else button.removeAttribute("aria-busy");
}
document.addEventListener("htmx:beforeRequest", (event) => markBusy(event, true));
document.addEventListener("htmx:afterRequest", (event) => markBusy(event, false));

// Space plays or pauses and the arrow keys move 5 s, unless the focus is in a field or on a control
// that uses those keys itself.
const STEP_S = 5;
document.addEventListener("keydown", (event) => {
  const player = document.getElementById("player");
  const here = event.target;
  if (!player || event.ctrlKey || event.altKey || event.metaKey) return;
  if (here.closest("input, textarea, select, button, summary, audio, a, [contenteditable]")) return;
  if (event.key === " ") {
    if (player.paused) player.play().catch(() => {});
    else player.pause();
  } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    const by = event.key === "ArrowLeft" ? -STEP_S : STEP_S;
    player.currentTime = Math.max(0, Math.min(player.duration || 0, player.currentTime + by));
  } else {
    return;
  }
  event.preventDefault();
});

// Live progress: the server sends the step list about once a second, a sentence when the step changes,
// and "done" when the analysis ends, which loads the page again with the results.
function follow(section) {
  const stream = new EventSource(section.dataset.events);
  const list = section.querySelector("[data-progress]");
  const now = section.querySelector("[data-now]");
  stream.addEventListener("progress", (event) => htmx.swap(list, event.data, { swapStyle: "innerHTML" }));
  stream.addEventListener("now", (event) => {
    now.textContent = event.data;
  });
  stream.addEventListener("done", () => {
    stream.close();
    window.location.reload();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const section = document.querySelector("[data-events]");
  if (section) follow(section);
});
