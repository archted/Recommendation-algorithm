[首页](https://www.jianshu.com/)[下载APP](https://www.jianshu.com/apps?utm_medium=desktop&utm_source=navbar-apps)[IT技术](https://www.jianshu.com/techareas)



# Markdown数学公式语法

[![img](https://cdn2.jianshu.io/assets/default_avatar/11-4d7c6ca89f439111aff57b23be1c73ba.jpg)DanielGavin](https://www.jianshu.com/u/b9e39a7a27ac)关注

# Markdown数学公式语法

[![img](https://cdn2.jianshu.io/assets/default_avatar/11-4d7c6ca89f439111aff57b23be1c73ba.jpg)](https://www.jianshu.com/u/b9e39a7a27ac)

[DanielGavin](https://www.jianshu.com/u/b9e39a7a27ac)关注

262018.02.09 19:56:08字数 2,117阅读 228,957

## 行内与独行

1. 行内公式：将公式插入到本行内，符号：`$公式内容$`，如：$xyz$
2. 独行公式：将公式插入到新的一行内，并且居中，符号：`$$公式内容$$`，如：$$xyz$$

## 上标、下标与组合

1. 上标符号，符号：`^`，如：$x^4$
2. 下标符号，符号：`_`，如：$x_1$
3. 组合符号，符号：`{}`，如：${16}_{8}O{2+}_{2}$

## 汉字、字体与格式

1. 汉字形式，符号：`\mbox{}`，如：$V_{\mbox{初始}}$
2. 字体控制，符号：`\displaystyle`，如：$\displaystyle \frac{x+y}{y+z}$
3. 下划线符号，符号：`\underline`，如：$\underline{x+y}$
4. 标签，符号`\tag{数字}`，如：$\tag{11}$
5. 上大括号，符号：`\overbrace{算式}`，如：$\overbrace{a+b+c+d}^{2.0}$
6. 下大括号，符号：`\underbrace{算式}`，如：$a+\underbrace{b+c}_{1.0}+d$
7. 上位符号，符号：`\stacrel{上位符号}{基位符号}`，如：$\vec{x}\stackrel{\mathrm{def}}{=}{x_1,\dots,x_n}$

## 占位符

1. 两个quad空格，符号：`\qquad`，如：$x \qquad y$
2. quad空格，符号：`\quad`，如：$x \quad y$
3. 大空格，符号`\`，如：$x \ y$
4. 中空格，符号`\:`，如：$x : y$
5. 小空格，符号`\,`，如：$x , y$
6. 没有空格，符号``，如：$xy$
7. 紧贴，符号`\!`，如：$x ! y$

## 定界符与组合

1. 括号，符号：`（）\big(\big) \Big(\Big) \bigg(\bigg) \Bigg(\Bigg)`，如：$（）\big(\big) \Big(\Big) \bigg(\bigg) \Bigg(\Bigg)$
2. 中括号，符号：`[]`，如：$[x+y]$
3. 大括号，符号：`\{ \}`，如：${x+y}$
4. 自适应括号，符号：`\left \right`，如：$\left(x\right)$，$\left(x{yz}\right)$
5. 组合公式，符号：`{上位公式 \choose 下位公式}`，如：${n+1 \choose k}={n \choose k}+{n \choose k-1}$
6. 组合公式，符号：`{上位公式 \atop 下位公式}`，如：$\sum_{k_0,k_1,\ldots>0 \atop k_0+k_1+\cdots=n}A_{k_0}A_{k_1}\cdots$

## 四则运算

1. 加法运算，符号：`+`，如：$x+y=z$
2. 减法运算，符号：`-`，如：$x-y=z$
3. 加减运算，符号：`\pm`，如：$x \pm y=z$
4. 减甲运算，符号：`\mp`，如：$x \mp y=z$
5. 乘法运算，符号：`\times`，如：$x \times y=z$
6. 点乘运算，符号：`\cdot`，如：$x \cdot y=z$
7. 星乘运算，符号：`\ast`，如：$x \ast y=z$
8. 除法运算，符号：`\div`，如：$x \div y=z$
9. 斜法运算，符号：`/`，如：$x/y=z$
10. 分式表示，符号：`\frac{分子}{分母}`，如：$\frac{x+y}{y+z}$
11. 分式表示，符号：`{分子} \voer {分母}`，如：${x+y} \over {y+z}$
12. 绝对值表示，符号：`||`，如：$|x+y|$

## 高级运算

1. 平均数运算，符号：`\overline{算式}`，如：$\overline{xyz}$
2. 开二次方运算，符号：`\sqrt`，如：$\sqrt x$
3. 开方运算，符号：`\sqrt[开方数]{被开方数}`，如：$\sqrt[3]{x+y}$
4. 对数运算，符号：`\log`，如：$\log(x)$
5. 极限运算，符号：`\lim`，如：$\lim^{x \to \infty}_{y \to 0}{\frac{x}{y}}$
6. 极限运算，符号：`\displaystyle \lim`，如：$\displaystyle \lim^{x \to \infty}_{y \to 0}{\frac{x}{y}}$
7. 求和运算，符号：`\sum`，如：$\sum^{x \to \infty}_{y \to 0}{\frac{x}{y}}$
8. 求和运算，符号：`\displaystyle \sum`，如：$\displaystyle \sum^{x \to \infty}_{y \to 0}{\frac{x}{y}}$
9. 积分运算，符号：`\int`，如：$\int^{\infty}_{0}{xdx}$
10. 积分运算，符号：`\displaystyle \int`，如：$\displaystyle \int^{\infty}_{0}{xdx}$
11. 微分运算，符号：`\partial`，如：$\frac{\partial x}{\partial y}$
12. 矩阵表示，符号：`\begin{matrix} \end{matrix}`，如：$\left[ \begin{matrix} 1 &2 &\cdots &4\5 &6 &\cdots &8\\vdots &\vdots &\ddots &\vdots\13 &14 &\cdots &16\end{matrix} \right]$

## 逻辑运算

1. 等于运算，符号：`=`，如：$x+y=z$
2. 大于运算，符号：`>`，如：$x+y>z$
3. 小于运算，符号：`<`，如：$x+y<z$
4. 大于等于运算，符号：`\geq`，如：$x+y \geq z$
5. 小于等于运算，符号：`\leq`，如：$x+y \leq z$
6. 不等于运算，符号：`\neq`，如：$x+y \neq z$
7. 不大于等于运算，符号：`\ngeq`，如：$x+y \ngeq z$
8. 不大于等于运算，符号：`\not\geq`，如：$x+y \not\geq z$
9. 不小于等于运算，符号：`\nleq`，如：$x+y \nleq z$
10. 不小于等于运算，符号：`\not\leq`，如：$x+y \not\leq z$
11. 约等于运算，符号：`\approx`，如：$x+y \approx z$
12. 恒定等于运算，符号：`\equiv`，如：$x+y \equiv z$

## 集合运算

1. 属于运算，符号：`\in`，如：$x \in y$
2. 不属于运算，符号：`\notin`，如：$x \notin y$
3. 不属于运算，符号：`\not\in`，如：$x \not\in y$
4. 子集运算，符号：`\subset`，如：$x \subset y$
5. 子集运算，符号：`\supset`，如：$x \supset y$
6. 真子集运算，符号：`\subseteq`，如：$x \subseteq y$
7. 非真子集运算，符号：`\subsetneq`，如：$x \subsetneq y$
8. 真子集运算，符号：`\supseteq`，如：$x \supseteq y$
9. 非真子集运算，符号：`\supsetneq`，如：$x \supsetneq y$
10. 非子集运算，符号：`\not\subset`，如：$x \not\subset y$
11. 非子集运算，符号：`\not\supset`，如：$x \not\supset y$
12. 并集运算，符号：`\cup`，如：$x \cup y$
13. 交集运算，符号：`\cap`，如：$x \cap y$
14. 差集运算，符号：`\setminus`，如：$x \setminus y$
15. 同或运算，符号：`\bigodot`，如：$x \bigodot y$
16. 同与运算，符号：`\bigotimes`，如：$x \bigotimes y$
17. 实数集合，符号：`\mathbb{R}`，如：`\mathbb{R}`
18. 自然数集合，符号：`\mathbb{Z}`，如：`\mathbb{Z}`
19. 空集，符号：`\emptyset`，如：$\emptyset$

## 数学符号

1. 无穷，符号：`\infty`，如：$\infty$
2. 虚数，符号：`\imath`，如：$\imath$
3. 虚数，符号：`\jmath`，如：$\jmath$
4. 数学符号，符号`\hat{a}`，如：$\hat{a}$
5. 数学符号，符号`\check{a}`，如：$\check{a}$
6. 数学符号，符号`\breve{a}`，如：$\breve{a}$
7. 数学符号，符号`\tilde{a}`，如：$\tilde{a}$
8. 数学符号，符号`\bar{a}`，如：$\bar{a}$
9. 矢量符号，符号`\vec{a}`，如：$\vec{a}$
10. 数学符号，符号`\acute{a}`，如：$\acute{a}$
11. 数学符号，符号`\grave{a}`，如：$\grave{a}$
12. 数学符号，符号`\mathring{a}`，如：$\mathring{a}$
13. 一阶导数符号，符号`\dot{a}`，如：$\dot{a}$
14. 二阶导数符号，符号`\ddot{a}`，如：$\ddot{a}$
15. 上箭头，符号：`\uparrow`，如：$\uparrow$
16. 上箭头，符号：`\Uparrow`，如：$\Uparrow$
17. 下箭头，符号：`\downarrow`，如：$\downarrow$
18. 下箭头，符号：`\Downarrow`，如：$\Downarrow$
19. 左箭头，符号：`\leftarrow`，如：$\leftarrow$
20. 左箭头，符号：`\Leftarrow`，如：$\Leftarrow$
21. 右箭头，符号：`\rightarrow`，如：$\rightarrow$
22. 右箭头，符号：`\Rightarrow`，如：$\Rightarrow$
23. 底端对齐的省略号，符号：`\ldots`，如：$1,2,\ldots,n$
24. 中线对齐的省略号，符号：`\cdots`，如：$x_1^2 + x_2^2 + \cdots + x_n^2$
25. 竖直对齐的省略号，符号：`\vdots`，如：$\vdots$
26. 斜对齐的省略号，符号：`\ddots`，如：$\ddots$

## 希腊字母

| 字母 | 实现       | 字母 | 实现       |
| ---- | ---------- | ---- | ---------- |
| A    | `A`        | α    | `\alhpa`   |
| B    | `B`        | β    | `\beta`    |
| Γ    | `\Gamma`   | γ    | `\gamma`   |
| Δ    | `\Delta`   | δ    | `\delta`   |
| E    | `E`        | ϵ    | `\epsilon` |
| Z    | `Z`        | ζ    | `\zeta`    |
| H    | `H`        | η    | `\eta`     |
| Θ    | `\Theta`   | θ    | `\theta`   |
| I    | `I`        | ι    | `\iota`    |
| K    | `K`        | κ    | `\kappa`   |
| Λ    | `\Lambda`  | λ    | `\lambda`  |
| M    | `M`        | μ    | `\mu`      |
| N    | `N`        | ν    | `\nu`      |
| Ξ    | `\Xi`      | ξ    | `\xi`      |
| O    | `O`        | ο    | `\omicron` |
| Π    | `\Pi`      | π    | `\pi`      |
| P    | `P`        | ρ    | `\rho`     |
| Σ    | `\Sigma`   | σ    | `\sigma`   |
| T    | `T`        | τ    | `\tau`     |
| Υ    | `\Upsilon` | υ    | `\upsilon` |
| Φ    | `\Phi`     | ϕ    | `\phi`     |
| X    | `X`        | χ    | `\chi`     |
| Ψ    | `\Psi`     | ψ    | `\psi`     |
| Ω    | `\v`       | ω    | `\omega`   |



250人点赞



[笔记](https://www.jianshu.com/nb/22124413)



[![  ](https://cdn2.jianshu.io/assets/default_avatar/11-4d7c6ca89f439111aff57b23be1c73ba.jpg)](https://www.jianshu.com/u/b9e39a7a27ac)

[DanielGavin](https://www.jianshu.com/u/b9e39a7a27ac)

总资产84共写了8248字获得261个赞共31个粉丝

关注

### 精彩评论1

[![img](https://upload.jianshu.io/users/upload_avatars/7822237/2419f321-306a-4707-bb64-3b0fcdd7f90f.png?imageMogr2/auto-orient/strip|imageView2/1/w/80/h/80/format/webp)](https://www.jianshu.com/u/791d3187121d)

[闪电的蓝熊猫](https://www.jianshu.com/u/791d3187121d)

2楼 2018.12.11 16:49

显然简书不支持这些东西

 9 回复

### 全部评论19只看作者按时间倒序按时间正序

[![img](https://upload.jianshu.io/users/upload_avatars/14371593/ff39bab3-8673-4040-b4ba-973474dc01a2.jpeg?imageMogr2/auto-orient/strip|imageView2/1/w/80/h/80/format/webp)](https://www.jianshu.com/u/82620c45aa7b)

[星语star](https://www.jianshu.com/u/82620c45aa7b)

23楼 2020.11.30 21:58

https://www.codecogs.com/latex/eqneditor.php?lang=zh-cn
简书写公式这个快点

 赞 回复

[![img](https://upload.jianshu.io/users/upload_avatars/14371593/ff39bab3-8673-4040-b4ba-973474dc01a2.jpeg?imageMogr2/auto-orient/strip|imageView2/1/w/80/h/80/format/webp)](https://www.jianshu.com/u/82620c45aa7b)

[星语star](https://www.jianshu.com/u/82620c45aa7b)

22楼 2020.11.30 21:52

\bar{ }在简书显示的是平方，不是平均数，有没有小伙伴知道怎么写出字母平均数上面的一杆？

 赞 回复

[![img](https://upload.jianshu.io/users/upload_avatars/14371593/ff39bab3-8673-4040-b4ba-973474dc01a2.jpeg?imageMogr2/auto-orient/strip|imageView2/1/w/80/h/80/format/webp)](https://www.jianshu.com/u/82620c45aa7b)

[星语star](https://www.jianshu.com/u/82620c45aa7b)

2020.11.30 21:56

\overline{}

 回复

[![img](https://cdn2.jianshu.io/assets/default_avatar/11-4d7c6ca89f439111aff57b23be1c73ba.jpg)](https://www.jianshu.com/u/b9e39a7a27ac)

[DanielGavin](https://www.jianshu.com/u/b9e39a7a27ac)作者

03.08 21:47

@湖红点鲑 \overline

 回复

 添加新评论

[![img](https://upload.jianshu.io/users/upload_avatars/14371593/ff39bab3-8673-4040-b4ba-973474dc01a2.jpeg?imageMogr2/auto-orient/strip|imageView2/1/w/80/h/80/format/webp)](https://www.jianshu.com/u/82620c45aa7b)

[星语star](https://www.jianshu.com/u/82620c45aa7b)

21楼 2020.11.30 21:49

https://www.codecogs.com/latex/eqneditor.php?lang=zh-cn
写公式这个快点

 赞 回复

[![img](https://upload.jianshu.io/users/upload_avatars/14371593/ff39bab3-8673-4040-b4ba-973474dc01a2.jpeg?imageMogr2/auto-orient/strip|imageView2/1/w/80/h/80/format/webp)](https://www.jianshu.com/u/82620c45aa7b)

[星语star](https://www.jianshu.com/u/82620c45aa7b)

20楼 2020.11.30 21:46

符号：后面的代码，应该用两个美元符号围起来，而不是用两个`

 赞 回复

[![img](https://upload.jianshu.io/users/upload_avatars/24413538/86bb05f1-82ee-45ea-b8b5-4654de13b733?imageMogr2/auto-orient/strip|imageView2/1/w/80/h/80/format/webp)](https://www.jianshu.com/u/d3b97160c6dd)

[良木66](https://www.jianshu.com/u/d3b97160c6dd)

18楼 2020.09.03 20:58

可以转载吗

 赞 回复

[![img](https://cdn2.jianshu.io/assets/default_avatar/13-394c31a9cb492fcb39c27422ca7d2815.jpg)](https://www.jianshu.com/u/fc9f4a446a93)

[夜伴轻雨](https://www.jianshu.com/u/fc9f4a446a93)

17楼 2020.06.09 19:37

感谢楼主的分享，我能做成脑图的形式分享出去吗

 赞 回复

[![img](https://cdn2.jianshu.io/assets/default_avatar/13-394c31a9cb492fcb39c27422ca7d2815.jpg)](https://www.jianshu.com/u/85ae589bfb02)

[85ae589bfb02](https://www.jianshu.com/u/85ae589bfb02)

10楼 2020.03.08 16:25

请问一下，这篇文章我能够转载吗

 赞 回复

[![img](https://cdn2.jianshu.io/assets/default_avatar/8-a356878e44b45ab268a3b0bbaaadeeb7.jpg)](https://www.jianshu.com/u/6212ed9ceef4)

[Springlord888](https://www.jianshu.com/u/6212ed9ceef4)

9楼 2019.12.31 10:23

希腊字母最后一行的大欧书写有误，应该为 \Omega

 1 回复

[![img](https://cdn2.jianshu.io/assets/default_avatar/8-a356878e44b45ab268a3b0bbaaadeeb7.jpg)](https://www.jianshu.com/u/6212ed9ceef4)

[Springlord888](https://www.jianshu.com/u/6212ed9ceef4)

8楼 2019.12.31 10:11

很齐全的总结

 赞 回复

[![img](https://upload.jianshu.io/users/upload_avatars/1935121/b97b710b-82d1-43b7-a708-aa2b46358a63.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/80/h/80/format/webp)](https://www.jianshu.com/u/85ca183ffbf5)

[培根炒蛋](https://www.jianshu.com/u/85ca183ffbf5)

7楼 2019.06.02 21:47

在我这里看你的公式全都挂了....

 1 回复

[![img](https://cdn2.jianshu.io/assets/default_avatar/6-fd30f34c8641f6f32f5494df5d6b8f3c.jpg)](https://www.jianshu.com/u/8c8c08331b81)

[8c8c08331b81](https://www.jianshu.com/u/8c8c08331b81)

2019.06.30 20:16

哈哈哈哈哈![:joy:](https://static.jianshu.io/assets/emojis/joy.png)

 回复

 添加新评论

- 1
- 2
- 下一页

### 被以下专题收入，发现更多相似内容

[![img](https://upload.jianshu.io/collections/images/1784680/tensorflow%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5%E6%9E%B6%E6%9E%84%E5%9B%BE.jpeg?imageMogr2/auto-orient/strip|imageView2/1/w/48/h/48/format/webp)dqn](https://www.jianshu.com/c/7e3ae9b0cae9)[![img](https://upload.jianshu.io/collections/images/1937055/file_5432777.png?imageMogr2/auto-orient/strip|imageView2/1/w/48/h/48/format/webp)网摘](https://www.jianshu.com/c/6060ae2cc0a7)[![img](https://upload.jianshu.io/collections/images/1923212/%E7%94%B5%E8%B7%AF_1.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/48/h/48/format/webp)other](https://www.jianshu.com/c/64e11017cac7)[![img](https://upload.jianshu.io/collections/images/1885515/psb_(1)_-_%E5%89%AF%E6%9C%AC.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/48/h/48/format/webp)latex](https://www.jianshu.com/c/e971afa1595d)[![img](https://upload.jianshu.io/collections/images/1807404/1564919130.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/48/h/48/format/webp)万花筒](https://www.jianshu.com/c/23fdeb234df1)[![img](https://upload.jianshu.io/collections/images/1875149/spikey.zh.png?imageMogr2/auto-orient/strip|imageView2/1/w/48/h/48/format/webp)Wolfram 语言](https://www.jianshu.com/c/f3c341b717eb)[![img](https://upload.jianshu.io/collections/images/1650196/Koala.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/48/h/48/format/webp)数据分析](https://www.jianshu.com/c/0defcfd01088)

展开更多

### 推荐阅读[更多精彩内容](https://www.jianshu.com/)

- [LaTeX学习](https://www.jianshu.com/p/5a22b53b11d7)

  $ \LaTeX{} $历史 $\LaTeX{}$（/ˈlɑːtɛx/，常被读作/ˈlɑːtɛk/或/ˈleɪtɛ...

  [![img](https://cdn2.jianshu.io/assets/default_avatar/8-a356878e44b45ab268a3b0bbaaadeeb7.jpg)大只若于](https://www.jianshu.com/u/557cc3da00e2)阅读 4,135评论 0赞 4

  ![img](https://upload-images.jianshu.io/upload_images/4378088-48960e30d3068ae7.png!web?imageMogr2/auto-orient/strip|imageView2/1/w/300/h/240/format/webp)

- [Markdown 添加 Latex 数学公式](https://www.jianshu.com/p/10925ddc6393)

  Markdown 中添加公式 行内公式 行间公式 Latex 数学公式语法 角标（上下标） 上标 下标 上下标命令...

  [![img](https://cdn2.jianshu.io/assets/default_avatar/10-e691107df16746d4a9f3fe9496fd1848.jpg)destiny0904](https://www.jianshu.com/u/a3e2567648b3)阅读 3,938评论 0赞 3

- [[转\]Mathjax, LaTex与Ghost](https://www.jianshu.com/p/5d347f37cb2d)

  声明！！！！ 此文章的代码部分在简书中皆不能正常显示， 请去我的个人网站观看效果, 如果访问不了, 请翻墙试试! ...

  [![img](https://cdn2.jianshu.io/assets/default_avatar/11-4d7c6ca89f439111aff57b23be1c73ba.jpg)kagenZhao](https://www.jianshu.com/u/ce91ccccd45c)阅读 1,418评论 0赞 0

- [Deformable Convolutional Networks论文翻译——中英文对照](https://www.jianshu.com/p/66b14a73f777)

  文章作者：Tyan博客：noahsnail.com | CSDN | 简书 声明：作者翻译论文仅为学习，如有侵权请...

  [![img](https://upload.jianshu.io/users/upload_avatars/3232548/242eb25c-3001-4411-a856-74b33312fdff.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/48/h/48/format/webp)SnailTyan](https://www.jianshu.com/u/7731e83f3a4e)阅读 3,845评论 1赞 3

  ![img](https://upload-images.jianshu.io/upload_images/3232548-0f6da4f8ebec0f03.png?imageMogr2/auto-orient/strip|imageView2/1/w/300/h/240/format/webp)

- [把自家婴儿打扮成哈利波特 红遍整个互联网](https://www.jianshu.com/p/6a8f858832e4)

  Lorelai Grace可能只是个普通的3个月大的婴儿，不过现在她已经因为“小哈利”新生儿照红遍网络了。她的母亲...

  [![img](https://cdn2.jianshu.io/assets/default_avatar/4-3397163ecdb3855a0a4139c34a695885.jpg)娱音社](https://www.jianshu.com/u/40a11c71948e)阅读 258评论 0赞 2

  ![img](https://upload-images.jianshu.io/upload_images/3498677-9715db90db53ae9f.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/300/h/240/format/webp)

[![img](https://cdn2.jianshu.io/assets/default_avatar/11-4d7c6ca89f439111aff57b23be1c73ba.jpg)](https://www.jianshu.com/u/b9e39a7a27ac)

[DanielGavin](https://www.jianshu.com/u/b9e39a7a27ac)

关注

总资产84

[机械运动](https://www.jianshu.com/p/2baab1628c6c)

阅读 228

### 推荐阅读

[输入输出格式化](https://www.jianshu.com/p/672e86586d77)

阅读 600

[LeetCode #640 Solve the Equation 求解方程](https://www.jianshu.com/p/50a995112640)

阅读 92

[计算机组成原理 2 数据的表示和运算 整数的表示](https://www.jianshu.com/p/8de0eefe57fd)

阅读 82

[SQL语句：常用函数](https://www.jianshu.com/p/2719c41f71c9)

阅读 162

[LeetCode-013-罗马数字转整数](https://www.jianshu.com/p/40d8f7063ed0)

阅读 69

评论19

赞250