document.addEventListener("DOMContentLoaded", () => {
  initPapers();
  initSidebarPlayer();
  initSectionNavigation();
});

async function initSidebarPlayer() {
  const audio = document.getElementById("sidebar-audio");
  const trackLabel = document.getElementById("player-track");
  const currentTime = document.getElementById("player-current");
  const duration = document.getElementById("player-duration");
  const progress = document.getElementById("player-progress");
  const previousButton = document.getElementById("player-previous");
  const toggleButton = document.getElementById("player-toggle");
  const nextButton = document.getElementById("player-next");
  const volume = document.getElementById("player-volume");

  if (!audio || !trackLabel || !currentTime || !duration || !progress || !previousButton || !toggleButton || !nextButton || !volume) return;

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

  function getTrackLabel(track) {
    return track.artist ? `${track.name} - ${track.artist}` : track.name;
  }

  function loadTrack(index, autoplay = false) {
    if (!playlist.length) return;
    currentIndex = (index + playlist.length) % playlist.length;
    const track = playlist[currentIndex];
    audio.pause();
    audio.src = track.url;
    audio.load();
    trackLabel.textContent = getTrackLabel(track);
    currentTime.textContent = "0:00";
    duration.textContent = "0:00";
    progress.value = "0";
    setPlayIcon(false);
    updateButtons();

    if (autoplay) {
      audio.play().catch(() => setPlayIcon(false));
    }
  }

  toggleButton.addEventListener("click", () => {
    if (currentIndex < 0) return;
    if (audio.paused) {
      if (audio.readyState === HTMLMediaElement.HAVE_NOTHING) audio.load();
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
  audio.addEventListener("error", () => {
    setPlayIcon(false);
    trackLabel.textContent = "Audio unavailable";
  });
  audio.addEventListener("ended", () => {
    if (playlist.length > 1) loadTrack(currentIndex + 1, true);
    else setPlayIcon(false);
  });

  try {
    const response = await fetch("/data/music.json?v=20260814-fixed-playlist");
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const data = await response.json();
    const publishedTracks = Array.isArray(data) ? data : data.tracks;

    if (Array.isArray(publishedTracks)) {
      publishedTracks.forEach(track => {
        if (!track || typeof track.src !== "string" || !track.src.trim()) return;
        playlist.push({
          name: String(track.title || track.name || "Untitled track"),
          artist: track.artist ? String(track.artist) : "",
          url: track.src.trim()
        });
      });
    }

    if (playlist.length) loadTrack(0);
  } catch (error) {}

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

  listContainer.addEventListener("click", event => {
    const trigger = event.target.closest("[data-paper-preview], [data-paper-cite]");
    if (!trigger) return;
    const paper = papers[Number(trigger.dataset.paperPreview ?? trigger.dataset.paperCite)];
    if (paper) openPaperPreview(paper, trigger.hasAttribute("data-paper-cite") ? "citation" : "overview");
  });
}

function renderPaper(paper, index) {
  const links = paper.links || {};
  const paperUrl = links.paper || links.pdf || "";
  const image = paper.image
    ? `<a class="paper-thumb" href="${escapeAttribute(paperUrl || paper.image)}" target="_blank" rel="noopener noreferrer">
         <img src="${escapeAttribute(paper.image)}" alt="${escapeAttribute(paper.title || "Publication image")}">
       </a>`
    : "";

  const actions = Object.entries(links)
    .filter(([, value]) => typeof value === "string" && value.trim())
    .map(([label, url]) => `<a href="${escapeAttribute(url)}" target="_blank" rel="noopener noreferrer"><i class="${label === "code" ? "fab fa-github" : "far fa-file-pdf"}" aria-hidden="true"></i>${escapeHtml(formatLabel(label))}</a>`)
    .join("");

  const tags = (paper.tags || [])
    .map(tag => `<span class="paper-meta-tag">${escapeHtml(tag)}</span>`)
    .join("");

  return `
    <article class="paper-item">
      ${image}
      <div class="paper-content">
        <span class="paper-badge">${escapeHtml(paper.venue || "")}</span>
        <h3 class="paper-title">${
          paperUrl
            ? `<a href="${escapeAttribute(paperUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(paper.title || "")}</a>`
            : escapeHtml(paper.title || "")
        }</h3>
        <p class="paper-authors">${highlightAuthorName(paper.authors || "")}</p>
        ${paper.summary ? `<p class="paper-summary">${escapeHtml(paper.summary)}</p>` : ""}
        ${tags ? `<div class="paper-meta-tags">${tags}</div>` : ""}
        <div class="paper-actions">${actions}<button type="button" data-paper-preview="${index}" aria-haspopup="dialog"><i class="far fa-eye" aria-hidden="true"></i>Preview</button><button type="button" class="citation-button" data-paper-cite="${index}" aria-label="Citation for ${escapeAttribute(paper.title)}" title="Citation" aria-haspopup="dialog"><i class="far fa-copy" aria-hidden="true"></i></button></div>
      </div>
    </article>
  `;
}

function citationFor(paper) {
  const authors = String(paper.authors || "").replace(/, and /g, ", ").split(", ").join(" and ");
  return `@inproceedings{li2026hyperpcn,\n  title = {${paper.title}},\n  author = {${authors}},\n  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},\n  year = {2026},\n  url = {${paper.links?.paper || ""}}\n}`;
}

function openPaperPreview(paper, initialTab = "overview") {
  const dialog = document.getElementById("paper-preview");
  const content = document.getElementById("preview-content");
  if (!dialog || !content) return;
  const citation = citationFor(paper);
  const paperUrl = escapeAttribute(paper.links?.paper || "");
  content.innerHTML = `
    <span class="paper-badge">${escapeHtml(paper.venue || "")}</span>
    <h2 id="preview-title">${escapeHtml(paper.title)}</h2>
    <p class="preview-authors">${highlightAuthorName(paper.authors)}</p>
    <div class="preview-tabs" role="tablist" aria-label="Publication details">
      ${[["overview", "Overview"], ["method", "Method"], ["paper", "PDF"], ["citation", "Cite"]].map(([id, label]) => `<button id="tab-${id}" type="button" role="tab" aria-controls="panel-${id}" aria-selected="false" tabindex="-1" data-preview-tab="${id}">${label}</button>`).join("")}
    </div>
    <section id="panel-overview" class="preview-panel" role="tabpanel" aria-labelledby="tab-overview" tabindex="0" hidden>
      <h3>Research Overview</h3><p>${escapeHtml(paper.summary || "")}</p>
      <p>${(paper.tags || []).map(tag => `<span class="paper-meta-tag">${escapeHtml(tag)}</span>`).join(" &middot; ")}</p>
      <figure class="preview-figure"><a href="${escapeAttribute(paper.image)}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size Hyper-PCN method figure"><img src="${escapeAttribute(paper.image)}" alt="Hyper-PCN framework overview"></a><figcaption>Hyper-PCN &middot; Framework overview</figcaption></figure>
    </section>
    <section id="panel-method" class="preview-panel" role="tabpanel" aria-labelledby="tab-method" tabindex="0" hidden><h3>Method Figure</h3><figure class="preview-figure"><a href="${escapeAttribute(paper.image)}" target="_blank" rel="noopener noreferrer" aria-label="Open full-size method figure"><img src="${escapeAttribute(paper.image)}" alt="Hyper-PCN encoder, feature embedding, refinement and decoder diagram"></a><figcaption><a href="${escapeAttribute(paper.image)}" target="_blank" rel="noopener noreferrer">Full-size figure <i class="fas fa-external-link-alt" aria-hidden="true"></i></a></figcaption></figure></section>
    <section id="panel-paper" class="preview-panel" role="tabpanel" aria-labelledby="tab-paper" tabindex="0" hidden><a class="preview-external" href="${paperUrl}" target="_blank" rel="noopener noreferrer"><i class="far fa-file-pdf" aria-hidden="true"></i> Open PDF in a new tab <i class="fas fa-external-link-alt" aria-hidden="true"></i></a><div id="preview-pdf-slot"></div></section>
    <section id="panel-citation" class="preview-panel" role="tabpanel" aria-labelledby="tab-citation" tabindex="0" hidden><h3>BibTeX</h3><pre class="preview-citation">${escapeHtml(citation)}</pre><button type="button" id="copy-citation" class="academic-icon-button" aria-label="Copy BibTeX citation" title="Copy BibTeX"><i class="far fa-copy" aria-hidden="true"></i></button><p class="citation-status" id="citation-status" role="status"></p></section>`;

  function selectTab(name, focus = false) {
    content.querySelectorAll("[data-preview-tab]").forEach(tab => {
      const active = tab.dataset.previewTab === name;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      if (active && focus) tab.focus();
    });
    content.querySelectorAll(".preview-panel").forEach(panel => { panel.hidden = panel.id !== `panel-${name}`; });
    if (name === "paper") {
      const slot = document.getElementById("preview-pdf-slot");
      if (!slot.hasChildNodes()) {
        const iframe = document.createElement("iframe");
        iframe.title = `${paper.title} PDF`;
        iframe.src = paper.links.paper;
        slot.appendChild(iframe);
      }
    }
  }

  content.querySelectorAll("[data-preview-tab]").forEach(tab => {
    tab.addEventListener("click", () => selectTab(tab.dataset.previewTab));
    tab.addEventListener("keydown", event => {
      const tabs = Array.from(content.querySelectorAll("[data-preview-tab]"));
      const index = tabs.indexOf(tab);
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); selectTab(tabs[next].dataset.previewTab, true); }
    });
  });
  content.querySelector("#copy-citation").addEventListener("click", async () => {
    const status = content.querySelector("#citation-status");
    try { await navigator.clipboard.writeText(citation); status.textContent = "Citation copied."; }
    catch { status.textContent = "Select and copy the citation above."; }
  });
  dialog.querySelector("[data-close-preview]").onclick = () => dialog.close();
  dialog.onclick = event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  };
  selectTab(initialTab);
  dialog.showModal();
  content.scrollTop = 0;
}

function initSectionNavigation() {
  if (!document.querySelector(".academic-home") || !("IntersectionObserver" in window)) return;
  const links = Array.from(document.querySelectorAll("#menus .site-page")).filter(link => link.hash);
  const observer = new IntersectionObserver(entries => {
    const entry = entries.filter(item => item.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!entry) return;
    links.forEach(link => {
      if (link.hash === `#${entry.target.id}`) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }, { rootMargin: "-80px 0px -45% 0px", threshold: [0, 0.3, 0.6] });
  document.querySelectorAll(".academic-home main > section").forEach(section => observer.observe(section));
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
