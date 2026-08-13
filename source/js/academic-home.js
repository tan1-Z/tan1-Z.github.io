document.addEventListener("DOMContentLoaded", () => {
  initPapers();
  initSidebarPlayer();
});

function initSidebarPlayer() {
  const audio = document.getElementById("sidebar-audio");
  const fileInput = document.getElementById("player-files");
  const trackLabel = document.getElementById("player-track");
  const currentTime = document.getElementById("player-current");
  const duration = document.getElementById("player-duration");
  const progress = document.getElementById("player-progress");
  const previousButton = document.getElementById("player-previous");
  const toggleButton = document.getElementById("player-toggle");
  const nextButton = document.getElementById("player-next");
  const volume = document.getElementById("player-volume");

  if (!audio || !fileInput || !trackLabel || !currentTime || !duration || !progress || !previousButton || !toggleButton || !nextButton || !volume) return;

  const playlist = [];
  let currentIndex = -1;
  let seeking = false;

  audio.volume = Number(volume.value);

  function formatTime(value) {
    if (!Number.isFinite(value) || value < 0) return "0:00";
    const minutes = Math.floor(value / 60);
    const seconds = Math.floor(value % 60).toString().padStart(2, "0");
    return `${minutes}:${seconds}`;
  }

  function setPlayIcon(isPlaying) {
    const icon = toggleButton.querySelector("i");
    if (!icon) return;
    icon.className = isPlaying ? "fas fa-pause" : "fas fa-play";
    toggleButton.setAttribute("aria-label", isPlaying ? "Pause" : "Play");
    toggleButton.title = isPlaying ? "Pause" : "Play";
  }

  function updateButtons() {
    const hasTrack = currentIndex >= 0;
    const hasMultipleTracks = playlist.length > 1;
    toggleButton.disabled = !hasTrack;
    progress.disabled = !hasTrack;
    previousButton.disabled = !hasMultipleTracks;
    nextButton.disabled = !hasMultipleTracks;
  }

  function loadTrack(index, autoplay = false) {
    if (!playlist.length) return;
    currentIndex = (index + playlist.length) % playlist.length;
    const track = playlist[currentIndex];
    audio.src = track.url;
    trackLabel.textContent = track.name;
    currentTime.textContent = "0:00";
    duration.textContent = "0:00";
    progress.value = "0";
    setPlayIcon(false);
    updateButtons();

    if (autoplay) {
      audio.play().catch(() => setPlayIcon(false));
    }
  }

  fileInput.addEventListener("change", () => {
    const files = Array.from(fileInput.files || []).filter(file => file.type.startsWith("audio/") || /\.(mp3|wav|ogg|m4a|aac|flac|opus)$/i.test(file.name));
    if (!files.length) return;

    const firstNewTrack = playlist.length;
    files.forEach(file => {
      playlist.push({
        name: file.name.replace(/\.[^.]+$/, ""),
        url: URL.createObjectURL(file)
      });
    });
    fileInput.value = "";
    loadTrack(firstNewTrack);
  });

  toggleButton.addEventListener("click", () => {
    if (currentIndex < 0) return;
    if (audio.paused) {
      audio.play().catch(() => setPlayIcon(false));
    } else {
      audio.pause();
    }
  });

  previousButton.addEventListener("click", () => loadTrack(currentIndex - 1, !audio.paused));
  nextButton.addEventListener("click", () => loadTrack(currentIndex + 1, !audio.paused));

  progress.addEventListener("input", () => {
    seeking = true;
    const targetTime = Number(progress.value);
    currentTime.textContent = formatTime(targetTime);
  });

  progress.addEventListener("change", () => {
    if (Number.isFinite(audio.duration)) audio.currentTime = Number(progress.value);
    seeking = false;
  });

  volume.addEventListener("input", () => {
    audio.volume = Number(volume.value);
  });

  audio.addEventListener("loadedmetadata", () => {
    progress.max = Number.isFinite(audio.duration) ? String(audio.duration) : "0";
    duration.textContent = formatTime(audio.duration);
  });

  audio.addEventListener("timeupdate", () => {
    if (seeking) return;
    progress.value = String(audio.currentTime || 0);
    currentTime.textContent = formatTime(audio.currentTime);
  });

  audio.addEventListener("play", () => setPlayIcon(true));
  audio.addEventListener("pause", () => setPlayIcon(false));
  audio.addEventListener("ended", () => {
    if (playlist.length > 1) loadTrack(currentIndex + 1, true);
    else setPlayIcon(false);
  });

  window.addEventListener("beforeunload", () => {
    playlist.forEach(track => URL.revokeObjectURL(track.url));
  });

  updateButtons();
}

async function initPapers() {
  const tagsContainer = document.getElementById("paper-tags");
  const listContainer = document.getElementById("paper-list");

  if (!listContainer) return;

  let papers = [];
  let currentTag = "All";

  try {
    const response = await fetch("/data/papers.json");
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    papers = await response.json();
  } catch (error) {
    listContainer.innerHTML = `<p>Publication data failed to load.</p>`;
    return;
  }

  const allTags = ["All", ...new Set(papers.flatMap(paper => paper.tags || []))];

  function renderTags() {
    if (!tagsContainer) return;

    tagsContainer.innerHTML = allTags
      .map(tag => {
        const active = tag === currentTag ? " active" : "";
        return `<button class="paper-tag${active}" type="button" data-tag="${escapeAttribute(tag)}">${escapeHtml(tag)}</button>`;
      })
      .join("");

    tagsContainer.querySelectorAll(".paper-tag").forEach(button => {
      button.addEventListener("click", () => {
        currentTag = button.dataset.tag;
        renderTags();
        renderList();
      });
    });
  }

  function renderList() {
    const filtered = currentTag === "All"
      ? papers
      : papers.filter(paper => (paper.tags || []).includes(currentTag));

    if (!filtered.length) {
      listContainer.innerHTML = "<p>No publications under this tag yet.</p>";
      return;
    }

    listContainer.innerHTML = filtered.map(renderPaper).join("");
  }

  renderTags();
  renderList();
}

function renderPaper(paper) {
  const links = paper.links || {};
  const paperUrl = links.paper || links.pdf || "";
  const image = paper.image
    ? `<a class="paper-thumb" href="${escapeAttribute(paperUrl || paper.image)}" target="_blank" rel="noopener noreferrer">
         <img src="${escapeAttribute(paper.image)}" alt="${escapeAttribute(paper.title || "Publication image")}">
       </a>`
    : "";

  const actions = Object.entries(links)
    .filter(([, value]) => typeof value === "string" && value.trim())
    .map(([label, url]) => `<a href="${escapeAttribute(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(formatLabel(label))}</a>`)
    .join("");

  const tags = (paper.tags || [])
    .map(tag => `<span class="paper-meta-tag">${escapeHtml(tag)}</span>`)
    .join("");

  return `
    <article class="paper-item">
      ${image}
      <div class="paper-content">
        <h3 class="paper-title">${
          paperUrl
            ? `<a href="${escapeAttribute(paperUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(paper.title || "")}</a>`
            : escapeHtml(paper.title || "")
        }</h3>
        <p class="paper-authors">${highlightAuthorName(paper.authors || "")}</p>
        <p class="paper-venue">${escapeHtml(paper.venue || "")}</p>
        ${actions ? `<div class="paper-actions">${actions}</div>` : ""}
        ${paper.summary ? `<p class="paper-summary">${escapeHtml(paper.summary)}</p>` : ""}
        ${tags ? `<div class="paper-meta-tags">${tags}</div>` : ""}
      </div>
    </article>
  `;
}

function formatLabel(label) {
  if (label.toLowerCase() === "pdf") return "PDF";
  return String(label).charAt(0).toUpperCase() + String(label).slice(1);
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

function highlightAuthorName(value) {
  let safe = escapeHtml(value || "");
  ["Pei Tan", "Tan Pei", "PeiTan"].forEach(name => {
    const escapedName = escapeHtml(name);
    safe = safe.replaceAll(escapedName, `<strong class="author-highlight">${escapedName}</strong>`);
  });
  return safe;
}
