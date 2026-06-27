document.addEventListener("DOMContentLoaded", () => {
  initPapers();
});

async function initPapers() {
  const tagsContainer = document.getElementById("paper-tags");
  const listContainer = document.getElementById("paper-list");

  if (!tagsContainer || !listContainer) return;

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
        <h3 class="paper-title">${escapeHtml(paper.title || "")}</h3>
        <p class="paper-authors">${highlightAuthorName(paper.authors || "")}</p>
        <p class="paper-venue">${escapeHtml(paper.venue || "")}</p>
        ${paper.summary ? `<p class="paper-summary">${escapeHtml(paper.summary)}</p>` : ""}
        ${actions ? `<div class="paper-actions">${actions}</div>` : ""}
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
