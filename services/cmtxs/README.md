# 听川好友房服务 v0.3.0

网页：https://tan1-z.github.io/cmtxs/rooms.html

在线服务：https://tingchuan-rooms.tingchuan-mahjong.workers.dev

备用入口：https://tingchuan-connect.pages.dev 。这是同一个房间服务的免费 Pages 转发入口，使用 `relay/_worker.js`；不另建牌局，也不复制状态。前端优先尝试备用入口，失败后尝试原地址，WebSocket 认证超时会切换入口。手机网络能否访问应实测 `/health`，不能仅凭电脑网络推断。

入座请求使用随机请求编号，创建与加入操作的超时重试返回原来的房间和座位。编号保留在当前浏览器会话，避免响应丢失后额外占座。兼容旧版访客凭证与已存在的房间。请求使用可清理的 `AbortController` 定时器，网络报错显示中文，保留表单并恢复按钮。

前端位于仓库 `source/cmtxs`。服务端位于 `services/cmtxs`，使用 Cloudflare Workers 与 SQLite Durable Objects；每个房间由一个 Durable Object 统一判定。账户当前使用 Workers Free；请保留免费套餐，额度耗尽时暂停服务，不自动开通付费。

## 运行与检查（GitHub 仓库目录）

需要 Node.js 24。

```sh
cd services/cmtxs
npm ci
npm test
npm run dev
```

访问 `http://127.0.0.1:4173/rooms.html`。两个浏览器页面即可联机，入口中可选择“作为新玩家加入”；刷新当前页恢复原座位。本地服务将房间写入 `.room-data`。原本的单机人机练习仍在首页。

```sh
npm run dev:cloudflare
node smoke.mjs http://127.0.0.1:8787
```

该检查会建立两个真实 WebSocket 连接，验证同桌入座、独立换牌与定缺、重连、重复请求、防泄露和完整一局结算。

## 部署与更新

```sh
npx wrangler login --scopes account:read user:read workers_scripts:write
npm run check
npm run deploy
```

本地开发项目 `outputs/tingchuan` 的对应命令为 `npm run deploy:rooms`，其配置在 `online/wrangler.jsonc`。首次部署的 Durable Object migration 使用 `new_sqlite_classes`，可在 Free 上运行。不要删除已有 migration 或房间命名空间。

## v0.3.0 玩法

单机的新开一局弹窗、好友房的房主大厅提供两个独立选项：普通血战/幺鸡万能牌、换三张/直接定缺。房主修改规则会清除所有真人的准备状态，开局后规则固定；重连保留本局规则，旧房间和旧牌谱默认为普通血战换三张。

幺鸡为 1 条，可代任意牌，可按条牌换出，可用于碰、直杠、暗杠或补杠；不能打出，不计入缺门牌。其他缺门牌须先清完才可碰杠胡。整副胡牌包括副露均无实际幺鸡时加 1 番，累计后五番封顶。胡牌按合法组成中最高番计分，查大叫同样使用本局规则。

副露保留 `tile`（声明牌面）与 `tiles`（实际用牌）两个字段。用幺鸡碰后补杠不会取回原幺鸡；带幺鸡的杠完成后，自己新摸到该杠对应的牌时可选择 `reclaim`，用这一张替换杠内一张幺鸡。该操作不额外补牌、不重复收杠分，保留手牌张数和 108 张实体牌守恒。前端提供明确用鸡数的碰杠按钮与换回按钮，服务端验证每项操作。

验证新组合：`node smoke.mjs <服务地址> --joker --no-swap`；移除 `--joker` 为普通血战，移除 `--no-swap` 为换三张。测试覆盖四种组合、过手胡、无鸡加番、缺门、带鸡碰杠和换回，以及持久化重连。

前端修改提交 `source/cmtxs` 后，由既有 Hexo Pages 工作流发布；`_config.yml` 的 `skip_render: cmtxs/**` 必须保留。服务端修改需要另外执行 Wrangler 部署。`source/cmtxs/online-config.js` 保存公开的服务地址。

本服务没有任何需要写进 GitHub 的私钥、Token 或密码；Wrangler 登录凭证由官方 CLI 在本机管理。禁止提交 `.room-data`、`.wrangler`、`.env` 等文件。

## 规则和一致性

- 1–4 位真人，空位补电脑；开局前等待真人准备。当前房主固定为第一个座位，换新牌友请创建新房间。
- 血战到底、同色换三张、同时定缺、自摸加番、五番封顶；查花猪、查大叫、未听退杠税；不启用呼叫转移。
- 每次操作 30 秒，超时自动托管，可取消托管。房间创建后保留 24 小时。
- 所有手牌、牌墙、计分均由服务端维护，客户端只提交操作；不会下发牌墙、随机种子、其他暗手或其他玩家凭证。
- 每个操作带唯一编号与决定编号；重复请求只确认一次，过时操作被拒绝。同一张弃牌收齐所有需要响应的座位后才判定，支持一炮多响。
- 过手胡保存在座位状态中；接炮需高于已过胡的封顶番数，自己的下一次摸牌（含杠后补牌）解除。
- 掉线保留座位，携带随机座位凭证恢复；再次打开同一座位会关闭旧连接。凭证不出现在邀请链接中。
- 对局及响应提交持久化后才发送确认；服务实例休眠和重启后从 SQLite 恢复，定时操作由 Durable Object alarm 驱动。WebSocket 使用 Hibernation API。
- 完局后提供自己的出牌观察与牌效复盘，不向同桌展示其他人的私有复盘。牌效按向听和未见有效进张计算，尚不代表长期得分最优。

## 当前范围

好友房使用浏览器访客座位，无账号系统、观战、匹配、聊天或跨设备找回凭证。清除站点数据后无法恢复原座位。仍在对局的房间不接收新玩家，想更换成员请另建房间。

测试覆盖 33 项：规则、四席输入验证、多人同时胡牌、过手胡、重复请求、非法/过时操作、各人数混合整局、隐私视图、真实 WebSocket 服务重启恢复，以及入座响应丢失、超时重试、跨入口恢复与重复占座保护。
