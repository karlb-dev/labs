const courseGrid = document.querySelector("#course-grid");

function courseVisual(kind, mark) {
  if (kind === "ring") {
    return `<div class="card-visual ring-visual" aria-hidden="true">
      <i class="ring-path"></i>
      <span class="rank r0">0</span><span class="rank r1">1</span><span class="rank r2">2</span><span class="rank r3">3</span>
      <b class="packet p1">→</b><b class="packet p2">→</b><b class="packet p3">→</b><b class="packet p4">→</b>
      <small>remote DMA · four-device ring</small>
    </div>`;
  }
  return `<div class="card-visual network-visual" aria-hidden="true">
    <i class="nv-line nl1"></i><i class="nv-line nl2"></i><i class="nv-line nl3"></i><i class="nv-line nl4"></i>
    <span class="nv-node nn1">tok</span><span class="nv-node nn2">L08</span><span class="nv-node nn3">L17</span><span class="nv-node nn4">L24</span><span class="nv-node nn5">out</span>
    <small>activation flow · causal intervention</small>
  </div>`;
}

courseGrid.innerHTML = COURSES.map(course => `
  <article class="course-card ${course.accent}">
    <div class="course-card-head"><span>${course.number} / ${course.track}</span><b>${course.mark}</b></div>
    ${courseVisual(course.visual, course.mark)}
    <div class="course-card-body">
      <h2>${course.title}</h2>
      <p>${course.description}</p>
      <div class="course-stats">${course.stats.map(stat => `<span>${stat}</span>`).join("")}</div>
      <div class="course-actions">
        <a class="button primary" href="${course.href}">Explore course <span>→</span></a>
        <a class="repo-mini" href="${course.repository}" target="_blank" rel="noreferrer">Source ↗</a>
      </div>
    </div>
  </article>`).join("");

document.querySelector("#course-total").textContent = String(COURSES.length).padStart(2, "0");
document.querySelector("#lab-total").textContent = COURSES.reduce((sum, course) => sum + Number.parseInt(course.stats[0], 10), 0);
