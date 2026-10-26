# 听川好友房服务 v0.2.0

网页：https://tan1-z.github.io/cmtxs/rooms.html

在线服务：https://tingchuan-rooms.tingchuan-mahjong.workers.dev

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

测试覆盖 28 项规则、四席输入验证、多人同时胡牌、过手胡、重复请求、非法/过时操作、各人数混合整局、隐私视图及真实 WebSocket 服务进程重启。
