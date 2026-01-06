(() => {
  const shapes = ["❤", "★", "✿", "❀"];
  const colors = ["#ff9eb7", "#7aa6ff", "#f7d794", "#c3a6ff"];
  let idx = 0;

  document.addEventListener("click", (e) => {
    const shape = shapes[idx % shapes.length];
    const color = colors[idx % colors.length];
    idx += 1;

    const el = document.createElement("span");
    const size = 14 + Math.random() * 12;
    el.textContent = shape;
    el.style.cssText = [
      "position:fixed",
      `left:${e.clientX}px`,
      `top:${e.clientY}px`,
      `font-size:${size}px`,
      `color:${color}`,
      "pointer-events:none",
      "transform:translate(-50%, -50%)",
      "transition: transform 1.2s ease, opacity 1.2s ease",
      "z-index:9999",
      "opacity:1",
    ].join(";");

    document.body.appendChild(el);

    requestAnimationFrame(() => {
      el.style.transform = "translate(-50%, -90px) scale(1.2)";
      el.style.opacity = "0";
    });

    setTimeout(() => el.remove(), 1200);
  });
})();
