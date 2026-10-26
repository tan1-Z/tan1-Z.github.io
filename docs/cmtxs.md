# 川麻听川研习室（cmtxs）

访问地址：https://tan1-z.github.io/cmtxs/

- 当前版本：听川 v0.1.1（包含过手胡修复）。
- 站点源码与静态资源位于 `source/cmtxs/`，无需额外打包或后端。
- Hexo 的 `skip_render` 包含 `cmtxs/**`，避免页面被博客主题改写。
- 资源使用相对路径，支持子目录部署。
- 推送到 main 后由现有 Pages 工作流生成并发布 `public/`。
- 牌面 SVG 的 M+ 授权原文保留在 `source/cmtxs/assets/tiles/LICENSE.txt`；上游为 https://github.com/lietxia/mahjong_graphic 。

## 更新与验证

更新 `source/cmtxs/` 中的对应文件，并运行 `npm ci`、`npm run build`。确认 `public/cmtxs/index.html`、JS/CSS 与全部 27 种牌面正常输出，再提交到 main。不要直接修改生成目录 public。

游戏数据保存在访问者浏览器中。原 ChatGPT Sites 域名下的牌谱可以导出 JSON，再在本页面导入。
