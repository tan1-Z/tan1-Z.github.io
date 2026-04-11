document.addEventListener("DOMContentLoaded", () => {
  initGameTabs();
  initSlotMachine();
  initMatchGame();
});

function initGameTabs() {
  const tabs = document.querySelectorAll(".game-tab");
  const panels = document.querySelectorAll(".game-panel");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.game;

      tabs.forEach(t => t.classList.remove("active"));
      panels.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      document.getElementById(`${target}-panel`).classList.add("active");
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

  if (!spinBtn || reels.includes(null) || !result) return;

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
    return arr.sort(() => Math.random() - 0.5);
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