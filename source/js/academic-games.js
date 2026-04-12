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
  const items = [
    "/img/game-icons/lulu-1.jpg",
    "/img/game-icons/lulu-2.jpg",
    "/img/game-icons/lulu-3.jpg",
    "/img/game-icons/lulu-4.jpg",
    "/img/game-icons/lulu-5.jpg",
    "/img/game-icons/lulu-6.jpg",
    "/img/game-icons/lulu-7.jpg",
    "/img/game-icons/lulu-8.jpg"
  ];

  const reels = [
    document.getElementById("slot-1"),
    document.getElementById("slot-2"),
    document.getElementById("slot-3")
  ];

  const spinBtn = document.getElementById("spin-btn");
  const result = document.getElementById("slot-result");

  if (!spinBtn || reels.includes(null) || !result) return;

  function getRandomItem() {
    return items[Math.floor(Math.random() * items.length)];
  }

  function renderItem(reel, item) {
    reel.innerHTML = `<img src="${item}" class="game-icon-img" alt="">`;
  }

  // 页面一加载就随机显示 3 个图标
  reels.forEach(reel => {
    renderItem(reel, getRandomItem());
  });

  spinBtn.addEventListener("click", () => {
    spinBtn.disabled = true;
    result.textContent = "正在抽取今日科研运势...";

    const finalValues = [];

    reels.forEach((reel, index) => {
      let count = 0;

      const timer = setInterval(() => {
        renderItem(reel, getRandomItem());
        count++;

        if (count > 15 + index * 6) {
          clearInterval(timer);

          const finalItem = getRandomItem();
          renderItem(reel, finalItem);
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

  const icons = [
    "/img/game-icons/lulu-1.jpg",
    "/img/game-icons/lulu-2.jpg",
    "/img/game-icons/lulu-3.jpg",
    "/img/game-icons/lulu-4.jpg",
    "/img/game-icons/lulu-5.jpg",
    "/img/game-icons/lulu-6.jpg",
    "/img/game-icons/lulu-7.jpg",
    "/img/game-icons/lulu-8.jpg"
  ];

  let firstCard = null;
  let secondCard = null;
  let lockBoard = false;
  let matchedCount = 0;

  function shuffle(arr) {
    return [...arr].sort(() => Math.random() - 0.5);
  }

  function startGame() {
    const pairs = shuffle([...icons, ...icons]);

    firstCard = null;
    secondCard = null;
    lockBoard = false;
    matchedCount = 0;
    grid.innerHTML = "";

    pairs.forEach((icon, idx) => {
      const cell = document.createElement("div");
      cell.className = "match-cell";
      cell.dataset.index = idx;
      cell.dataset.value = icon;

      cell.innerHTML = `
        <div class="match-card-inner">
          <div class="match-card-face match-card-front">
            <div class="match-card-front-mark"></div>
          </div>
          <div class="match-card-face match-card-back">
            <img src="${icon}" class="match-icon-img" alt="">
          </div>
        </div>
      `;

      cell.addEventListener("click", () => handleCardClick(cell));
      grid.appendChild(cell);
    });

    result.textContent = "一次翻两张，相同就消除，不同会翻回去。";
  }

  function handleCardClick(cell) {
    if (lockBoard) return;
    if (cell.classList.contains("flipped")) return;
    if (cell.classList.contains("matched")) return;

    cell.classList.add("flipped");

    if (!firstCard) {
      firstCard = cell;
      return;
    }

    secondCard = cell;
    lockBoard = true;

    const isMatch = firstCard.dataset.value === secondCard.dataset.value;

    if (isMatch) {
      setTimeout(() => {
        firstCard.classList.add("matched");
        secondCard.classList.add("matched");

        matchedCount += 2;

        if (matchedCount === icons.length * 2) {
          result.textContent = "全部配对完成！今天也要继续推进科研。";
        } else {
          result.textContent = "配对成功，继续加油。";
        }

        resetBoard();
      }, 350);
    } else {
      setTimeout(() => {
        firstCard.classList.remove("flipped");
        secondCard.classList.remove("flipped");
        result.textContent = "这两个不一样，再试一次。";
        resetBoard();
      }, 800);
    }
  }

  function resetBoard() {
    firstCard = null;
    secondCard = null;
    lockBoard = false;
  }

  resetBtn.addEventListener("click", startGame);
  startGame();
}