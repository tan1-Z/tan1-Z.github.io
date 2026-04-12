---
title: Academic
date: 2026-04-12 00:00:00
layout: page
top_img: false
aside: false
comments: false
---

<link rel="stylesheet" href="/css/academic-home.css">

<div class="academic-page">
<div class="academic-side academic-side-left">
<img src="/img/academic/side/left-1.jpg" alt="left-1">
<img src="/img/academic/side/left-2.jpg" alt="left-2">
<img src="/img/academic/side/left-3.jpg" alt="left-3">
<img src="/img/academic/side/left-4.jpg" alt="left-4">
<img src="/img/academic/side/left-5.jpg" alt="left-5">
<img src="/img/academic/side/left-6.jpg" alt="left-6">
</div>

<div class="academic-side academic-side-right">
<img src="/img/academic/side/right-1.jpg" alt="right-1">
<img src="/img/academic/side/right-2.jpg" alt="right-2">
<img src="/img/academic/side/right-3.jpg" alt="right-3">
<img src="/img/academic/side/right-4.jpg" alt="right-4">
<img src="/img/academic/side/right-5.jpg" alt="right-5">
<img src="/img/academic/side/right-6.jpg" alt="right-6">
</div>

<div class="academic-layout">
<aside class="academic-left">
<section class="academic-card profile-card">
<img class="profile-avatar" src="/img/academic/avatar.jpg" alt="Pei Tan">
<h1 class="profile-name">Pei Tan</h1>
<p class="profile-role">Undergraduate Student</p>
<p class="profile-school">School of Information and Software Engineering, UESTC</p>

<div class="profile-links">
<a href="https://github.com/tan1-Z" target="_blank">GitHub</a>
<a href="mailto:peit879@gmail.com">Email</a>
<a href="/about/">About</a>
</div>
</section>

<section class="academic-card game-card">
<div class="game-switch">
<button class="game-tab active" data-game="slot">老虎机</button>
<button class="game-tab" data-game="match">连连看</button>
</div>

<div class="game-panel active" id="slot-panel">
<div class="slot-machine">
<div class="slot-reel" id="slot-1"></div>
<div class="slot-reel" id="slot-2"></div>
<div class="slot-reel" id="slot-3"></div>
</div>
<button class="game-btn" id="spin-btn">开始抽奖</button>
<p class="game-result" id="slot-result">试试看今天的科研运势。</p>
</div>

<div class="game-panel" id="match-panel">
<div class="match-grid" id="match-grid"></div>
<button class="game-btn" id="reset-match-btn">重新开始</button>
<p class="game-result" id="match-result">找出所有相同图案并配对消除。</p>
</div>
</section>
</aside>

<main class="academic-main">
<section class="academic-card about-card">
<h2>About Me</h2>
<p>
Hello! I am Pei Tan (tan1), an undergraduate student from the School of Information and Software Engineering at UESTC.
My current interests include 3D vision, point cloud completion, and hypergraph learning.
</p>
<p>
This page is used to present my selected papers, projects, and ongoing work.
</p>
</section>

<section class="academic-card news-card">
<h2>News</h2>
<ul class="news-list">
<li><span>Feb 20, 2026</span> 🎉 One paper was accepted by CVPR 2026.</li>
</ul>
</section>

<section class="academic-card paper-card">
<div class="paper-header">
<h2>Papers</h2>
<p>Selected publications and research works</p>
</div>

<div class="paper-tags" id="paper-tags"></div>
<div class="paper-list" id="paper-list"></div>
</section>
</main>
</div>
</div>

<script src="/js/academic-home.js"></script>
