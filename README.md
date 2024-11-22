# hexo-blog


# FAQ

## MacOS 下安装 hexo 失败如何解决？

该项目在 MacOS M1/M2 下启动可能会报错，提示 `Chromium Binary Not Available for arm64`, 这是由于该项目依赖 `hexo-renderer-multi-markdown-it` 包，而这个依赖包依赖 puppeteer，puppeteer 依赖 chromium。

可以参考文章 [Resolving the “Chromium Binary Not Available for arm64” Error during Puppeteer Installation on M1/M2 Macs](https://luisrangelc.medium.com/resolving-the-chromium-binary-not-available-for-arm64-error-during-puppeteer-installation-on-12d02402f619), 使用 homebrew 手动安装 chromium，然后设置环境变量 `PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true` 跳过 chromium 的下载。


## 如何升级 hexo 版本

可以使用 `npm-check` 来升级 npm 项目的包依赖，`npm-check -u` 会列出所有需要升级的包，然后选择升级即可。
> npm-check 需要单独安装, 使用 `npm install -g npm-check` 安装即可。
