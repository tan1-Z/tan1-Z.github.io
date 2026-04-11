document.addEventListener("DOMContentLoaded", async () => {
  initGameTabs();
  initSlotMachine();
  initMatchGame();
  await initPapers();
});

function initGameTabs() {
  const tabs = document.querySelectorAll(".game-tab");
  const panels = document.querySelectorAll(".game-panel");

  if (!tabs.length || !panels.length) return;

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.game;

      tabs.forEach(t => t.classList.remove("active"));
      panels.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const panel = document.getElementById(`${target}-panel`);
      if (panel) panel.classList.add("active");
    });
  });
}

function initSlotMachine() {
  const items = ["🐸", "📚", "💻", "🧠", "✨", "☕", "🔥", "🎯"];
  const reels = [
    document.getElementById("slot-1"),
    document.getElementById("slot-2"),
    document.getElementById("slot-3")
  ];
  const spinBtn = document.getElementById("spin-btn");
  const result = document.getElementById("slot-result");

  if (!spinBtn || reels.some(r => !r) || !result) return;

  spinBtn.addEventListener("click", () => {
    spinBtn.disabled = true;
    result.textContent = "正在抽取今日科研运势...";

    const finalValues = [];

    reels.forEach((reel, index) => {
      let count = 0;

      const timer = setInterval(() => {
        reel.textContent = items[Math.floor(Math.random() * items.length)];
        count++;

        if (count > 15 + index * 6) {
          clearInterval(timer);
          const finalItem = items[Math.floor(Math.random() * items.length)];
          reel.textContent = finalItem;
          finalValues[index] = finalItem;

          if (finalValues.filter(Boolean).length === 3) {
            const [a, b, c] = finalValues;

            if (a === b && b === c) {
              result.textContent = "三连成功！今天状态爆棚，适合猛写论文。";
            } else if (a === b || b === c || a === c) {
              result.textContent = "小有好运！今天适合改实验和补细节。";
            } else {
              result.textContent = "平稳发挥的一天，适合推进日常任务。";
            }

            spinBtn.disabled = false;
          }
        }
      }, 80);
    });
  });
}

function initMatchGame() {
  const grid = document.getElementById("match-grid");
  const resetBtn = document.getElementById("reset-match-btn");
  const result = document.getElementById("match-result");

  if (!grid || !resetBtn || !result) return;

  const icons = ["🐸", "📘", "💻", "🧠", "🎓", "⭐", "📄", "☕"];
  let values = [];
  let selected = [];
  let clearedCount = 0;

  function shuffle(arr) {
    return [...arr].sort(() => Math.random() - 0.5);
  }

  function startGame() {
    const pairs = shuffle([...icons, ...icons]);
    values = pairs;
    selected = [];
    clearedCount = 0;
    grid.innerHTML = "";

    values.forEach((icon, idx) => {
      const cell = document.createElement("div");
      cell.className = "match-cell";
      cell.dataset.index = idx;
      cell.dataset.value = icon;
      cell.textContent = icon;

      cell.addEventListener("click", () => handleCellClick(cell));
      grid.appendChild(cell);
    });

    result.textContent = "找出所有相同图案并配对消除。";
  }

  function handleCellClick(cell) {
    if (cell.classList.contains("cleared")) return;
    if (selected.includes(cell)) return;
    if (selected.length >= 2) return;

    cell.classList.add("selected");
    selected.push(cell);

    if (selected.length === 2) {
      const [a, b] = selected;

      if (a.dataset.value === b.dataset.value) {
        setTimeout(() => {
          a.classList.remove("selected");
          b.classList.remove("selected");
          a.classList.add("cleared");
          b.classList.add("cleared");
          selected = [];
          clearedCount += 2;

          if (clearedCount === values.length) {
            result.textContent = "全部消除完成！今天也要继续推进科研。";
          } else {
            result.textContent = "配对成功，继续加油。";
          }
        }, 220);
      } else {
        setTimeout(() => {
          a.classList.remove("selected");
          b.classList.remove("selected");
          selected = [];
          result.textContent = "这两个不一样，再试一次。";
        }, 320);
      }
    }
  }

  resetBtn.addEventListener("click", startGame);
  startGame();
}

async function initPapers() {
  const tagsContainer = document.getElementById("paper-tags");
  const listContainer = document.getElementById("paper-list");

  if (!tagsContainer || !listContainer) return;

  let papers = [];
  let currentTag = "All";

  try {
    const res = await fetch("/data/papers.json");

    if (!res.ok) {
      throw new Error(`HTTP ${res.status} ${res.statusText}`);
    }

    papers = await res.json();
  } catch (err) {
    console.error("Failed to load papers.json:", err);
    listContainer.innerHTML = `<p>Paper data failed to load: ${escapeHtml(err.message)}</p>`;
    return;
  }

  const allTags = new Set(["All"]);
  papers.forEach(paper => {
    (paper.tags || []).forEach(tag => allTags.add(tag));
  });

  function renderTags() {
    tagsContainer.innerHTML = "";

    [...allTags].forEach(tag => {
      const btn = document.createElement("button");
      btn.className = "paper-tag" + (tag === currentTag ? " active" : "");
      btn.textContent = tag;

      btn.addEventListener("click", () => {
        currentTag = tag;
        renderTags();
        renderList();
      });

      tagsContainer.appendChild(btn);
    });
  }

  function renderList() {
    const filtered =
      currentTag === "All"
        ? papers
        : papers.filter(p => (p.tags || []).includes(currentTag));

    if (!filtered.length) {
      listContainer.innerHTML = "<p>No papers under this tag yet.</p>";
      return;
    }

    listContainer.innerHTML = filtered
      .map(paper => {
        const links = paper.links || {};

        const linkButtons = Object.entries(links)
          .filter(([_, value]) => typeof value === "string" && value.trim() !== "")
          .map(([key, value]) => {
            return `<a href="${escapeAttribute(value)}" target="_blank" rel="noopener noreferrer">${capitalize(key)}</a>`;
          })
          .join("");

        const metaTags = (paper.tags || [])
          .map(tag => `<span class="paper-meta-tag">${escapeHtml(tag)}</span>`)
          .join("");

        const thumbHtml = paper.image
          ? `
            <div class="paper-thumb">
              ${paper.badge ? `<div class="paper-thumb-badge">${escapeHtml(paper.badge)}</div>` : ""}
              ${
                links.paper
                  ? `<a href="${escapeAttribute(links.paper)}" target="_blank" rel="noopener noreferrer">
                       <img src="${escapeAttribute(paper.image)}" alt="${escapeAttribute(paper.title)}">
                     </a>`
                  : `<img src="${escapeAttribute(paper.image)}" alt="${escapeAttribute(paper.title)}">`
              }
            </div>
          `
          : "";

        return `
          <article class="paper-item">
            <div class="paper-item-inner">
              ${thumbHtml}
              <div class="paper-content">
                <div class="paper-title">${escapeHtml(paper.title || "")}</div>
                <div class="paper-authors">${highlightAuthorName(paper.authors || "")}</div>
                <div class="paper-venue">${escapeHtml(paper.venue || "")}</div>
                ${paper.summary ? `<div class="paper-summary">${escapeHtml(paper.summary)}</div>` : ""}
                ${linkButtons ? `<div class="paper-actions">${linkButtons}</div>` : ""}
                ${metaTags ? `<div class="paper-meta-tags">${metaTags}</div>` : ""}
              </div>
            </div>
          </article>
        `;
      })
      .join("");
  }

  renderTags();
  renderList();
}

function capitalize(str) {
  return String(str).charAt(0).toUpperCase() + String(str).slice(1);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeAttribute(str) {
  return String(str).replace(/"/g, "&quot;");
}

function highlightAuthorName(str) {
  let safe = escapeHtml(str || "");
  const targetNames = ["Pei Tan", "PeiTan", "Tan Pei"];

  targetNames.forEach(name => {
    const safeName = escapeHtml(name);
    safe = safe.replaceAll(
      safeName,
      `<strong class="author-highlight">${safeName}</strong>`
    );
  });

  return safe;
}