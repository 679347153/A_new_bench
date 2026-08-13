下面我把它整理成一套可以直接作为论文 **Method** 主体继续修改的完整方法论。这里不再围绕已有数据集展开，而是从你的新项目本身出发，把问题定义成：

> **在同一家庭场景中，对长期共同生活的多名居民进行多时间尺度行为建模，并利用日常活动、周期性行为、特殊事件及物体生命周期共同驱动家庭环境在一个月内连续演化，最终生成任意时间点对应的完整物体分布状态。**

核心目标不是生成“随机变化的房间”，而是生成一个具有**稳定规律、短期随机性、长期变化和事件因果关系**的家庭世界，使机器人能够通过多次访问同一场景逐步习得：

[
P(\text{object state/location}\mid
\text{time, history, household})
]

以及这种规律随时间发生变化时如何更新已有记忆。

---

# Method

## 1. Overview

我们提出一个 **Long-Horizon Household Dynamics Generation Framework**，用于生成具有月尺度时间跨度的长期动态家庭环境。对于给定的静态住宅场景，系统首先建立一组长期共同生活的居民，并为整个家庭初始化稳定的 household profile，包括居民的职业、作息、物品使用偏好、房间归属、共享关系、购物习惯、食品消耗习惯以及整理习惯等。

随后，生成器以“月—周—日—活动”四层时间结构构建居民行为。一方面，居民每天重复具有一定规律性的 daily routines，例如起床、洗漱、早餐、工作、晚餐、休闲和睡觉；另一方面，在一个月尺度下，系统额外模拟购物、清洁、访客、聚会、节日、物品损坏、新物品购买以及物品消耗等长期事件。

所有活动和事件最终都被转化为对环境状态的显式修改，包括：

[
\text{Move},
\quad
\text{Consume},
\quad
\text{Replenish},
\quad
\text{Introduce},
\quad
\text{Remove},
\quad
\text{Damage},
\quad
\text{Repair},
\quad
\text{Replace}.
]

与独立生成每一天不同，整个一个月采用**连续状态传播**：

[
S_{t_0}
\xrightarrow{e_1}
S_{t_1}
\xrightarrow{e_2}
S_{t_2}
\rightarrow \cdots
\xrightarrow{e_N}
S_{t_N},
]

因此 Day 15 的场景状态严格建立在 Day 1–14 已经发生的所有行为和事件之上。

整个生成系统可以概括为

[
\boxed{
\begin{aligned}
&\text{Static House}
+\text{Persistent Residents}
+\text{Geo-temporal Context}
\
&\qquad\downarrow\
&\text{Household Profile Initialization}
\
&\qquad\downarrow\
&\text{Monthly Event Planning}
\
&\qquad\downarrow\
&\text{Weekly Routine Modeling}
\
&\qquad\downarrow\
&\text{Daily Schedule Generation}
\
&\qquad\downarrow\
&\text{Fine-grained Indoor Activities}
\
&\qquad\downarrow\
&\text{Activity/Event State Transition}
\
&\qquad\downarrow\
&\text{Long-term Object Lifecycle Simulation}
\
&\qquad\downarrow\
&\text{Scene Grounding and Physical Validation}
\
&\qquad\downarrow\
&{S_{t_1},S_{t_2},\ldots,S_{t_K}}.
\end{aligned}}
]

---

# 2. Long-Horizon Dynamic Household Definition

给定时间区间

[
[t_s,t_e],
]

其中

[
t_e-t_s\approx 1\text{ month},
]

定义一个长期动态家庭场景为

[
\mathcal S=
(
\mathcal H,
\mathcal L,
\mathcal R,
\mathcal U,
\mathcal O,
\mathcal X,
\mathcal E,
\Theta,
\mathcal C,
t_s,t_e
).
]

其中：

| 符号              | 含义                        |
| --------------- | ------------------------- |
| (\mathcal H)    | 房屋固定几何结构                  |
| (\mathcal L)    | 房间集合                      |
| (\mathcal R)    | receptacle / furniture 集合 |
| (\mathcal U)    | 长期居民集合                    |
| (\mathcal O)    | 在整个模拟期间可能出现过的物体实例         |
| (\mathcal X(t)) | 时间 (t) 下环境完整状态            |
| (\mathcal E)    | 一个月内发生的活动和事件              |
| (\Theta)        | 家庭长期 latent profile       |
| (\mathcal C(t)) | 时间、月份、地点、日历等 context      |

对于任意时间 (t)，场景状态定义为

[
S_t=
(
\mathcal H,
\mathcal L,
\mathcal R,
\mathcal U,
\mathcal O_t,
\mathcal X_t
).
]

其中房屋结构在一个月内保持不变，而居民活动和长期事件不断改变

[
\mathcal O_t,\mathcal X_t.
]

换言之：

[
\mathcal H_t=\mathcal H,
]

但

[
\mathcal X_t\neq\mathcal X_{t+\Delta t}.
]

---

# 3. Object-Centric Environment State

为了描述长期变化，单纯记录 object pose 不够。每个物体实例 (o) 被赋予一个长期状态：

[
x_o(t)=
(
b_o,
c_o,
l_o,
r_o,
p_o,
\omega_o,
h_o,
u_o,
t_o^{last}
).
]

其中：

| 属性            | 含义                   |
| ------------- | -------------------- |
| (b_o(t))      | 是否当前存在于房屋中           |
| (c_o)         | object category      |
| (l_o(t))      | 当前所在房间               |
| (r_o(t))      | 当前所在 receptacle      |
| (p_o(t))      | 3D position          |
| (\omega_o(t)) | rotation             |
| (h_o(t))      | condition/state      |
| (u_o)         | owner / primary user |
| (t_o^{last})  | 最近一次发生状态变化的时间        |

其中

[
b_o(t)\in{0,1}.
]

因此已经损坏并丢弃的物品虽然仍属于历史对象集合

[
o\in\mathcal O,
]

但在当前场景中：

[
b_o(t)=0.
]

对于可消耗类别 (c)，进一步定义库存：

[
Q_c(t)
======

\sum_{o\in\mathcal O}
\mathbb I[c_o=c]
\mathbb I[b_o(t)=1].
]

例如：

[
Q_{\text{Apple}}(t)=6.
]

早餐消费一个苹果后：

[
Q_{\text{Apple}}(t+1)=5.
]

这样既可以在高层表达“苹果剩几个”，又可以在实际仿真中将每一个 Apple 建模成独立 3D 实例。

---

# 4. Multi-Time-Scale Household Dynamics

长期家庭环境包含明显不同的时间尺度。我们将其划分为 fast、medium 和 slow dynamics。

| Dynamics |  时间尺度 | 典型现象              |
| -------- | ----: | ----------------- |
| Fast     | 分钟–小时 | 手机、杯子、餐具、遥控器移动    |
| Medium   |  天–星期 | 食物消耗、购物、洗衣、清洁     |
| Slow     |  星期–月 | 新物品、损坏、丢弃、更换、节日变化 |

于是整个状态演化可以抽象为

[
X_{t+\Delta t}
==============

F_{\mathrm{fast}}
\circ
F_{\mathrm{medium}}
\circ
F_{\mathrm{slow}}
(X_t).
]

Fast dynamics 主要产生**位置分布规律**。

例如同一个人的 Coffee Mug 在早餐后大概率位于 kitchen/dining area，而晚上工作时可能位于 desk。

Medium dynamics 产生**数量和周期状态变化**。

例如：

[
Apple:
8\rightarrow7\rightarrow5\rightarrow3
\rightarrow1\rightarrow0
\rightarrow8.
]

Slow dynamics 则使场景本身出现 distribution shift，例如：

[
CoffeeMachine:
Absent
\rightarrow
Introduced
\rightarrow
FrequentlyUsed.
]

---

# 5. Persistent Household Profile

为了让机器人能够通过长期探索真正学到规律，场景动态不能每天独立随机生成。

我们为每个家庭采样一次长期 latent household profile：

[
\Theta_H=
{
\Theta_U,
\Theta_R,
\Theta_P,
\Theta_I,
\Theta_S,
\Theta_E
}.
]

其中包括居民属性、routine、placement preference、inventory habit、shopping habit 和 special-event propensity。

例如某家庭可以具有以下稳定特征：

| Household property | 示例                    |
| ------------------ | --------------------- |
| Breakfast time     | 工作日约 7:20             |
| Grocery shopping   | 通常周六上午                |
| Cleaning           | 周日上午                  |
| Resident A         | 经常在书房工作               |
| Resident B         | 喜欢在沙发上看电视             |
| Mug habit          | A 常把自己的杯子留在 desk      |
| Phone habit        | B 回家常把手机放在 side table |
| Tidiness           | 家庭整体中等偏整洁             |
| Apple usage        | 平均每日消耗约 1–2 个         |
| Shared object      | TV remote 被多人使用       |

这些参数在整个一个月中保持基本稳定。

因此：

[
P(X_t)
\neq
P(X_t\mid t)
]

而应表示成：

[
P(
X_t
\mid
\Theta_H,
C_t,
X_{<t},
E_{<t}
).
]

这也是整个数据集产生“可学习长期规律”的根本来源。

---

# 6. Multi-Resident Household Modeling

设长期居住居民集合为

[
\mathcal U=
{u_1,u_2,\ldots,u_N}.
]

每一个居民拥有自己的 profile：

[
\theta_i=
(
Age,
Occupation,
Routine,
Preference,
Ownership,
RoomAssignment,
Tidiness,
ConsumptionHabit
).
]

其中 RoomAssignment 用于建立稳定的个人空间。例如：

[
u_i\rightarrow Bedroom_i.
]

卫生间可以是私人或共享。

厨房、living room、dining room 等通常建模为 shared spaces。

## 6.1 Personal Objects

个人物体包括：

[
Phone_i,\ Laptop_i,\ Glasses_i,\ Bag_i,\ Mug_i.
]

对于个人物体：

[
P(
u_i\text{ uses }o_i
)
\gg
P(
u_j\text{ uses }o_i
),
\quad i\neq j.
]

因此物体位置可以携带明显的 resident-specific pattern。

---

## 6.2 Shared Objects

例如：

[
TVRemote,\ VacuumCleaner,\ Plates,
KitchenUtensils.
]

多人均可以改变其位置。

于是可能产生：

[
Sofa
\xrightarrow{u_1}
CoffeeTable
\xrightarrow{u_2}
TVStand.
]

因此 shared object 往往具有更高的位置熵，但仍受到家庭活动习惯约束。

---

## 6.3 Joint Activities

家庭活动并不全部属于单一居民。

允许一个事件对应居民子集：

[
U_e\subseteq\mathcal U.
]

例如：

[
{u_1,u_2,u_3}
\xrightarrow{\text{Family Dinner}}
DiningRoom.
]

这类活动可以同时调用：

[
Plate,\ Fork,\ Cup,\ Food,\ Napkin,
DiningChair.
]

因此，多居民之间不仅是 schedule 的简单叠加，而会形成真正的 household-level interactions。

---

# 7. Geo-Temporal Context

每一天引入 context：

[
C_d=
(
Location,
Month,
Date,
Weekday,
Season,
Calendar
).
]

当前默认：

[
Location=\text{Los Angeles, USA}.
]

月份：

[
Month\in{1,\ldots,12}.
]

Location 和 Month 不直接强制某个事件发生，而是影响：

[
P(e\mid C_d,\Theta_H).
]

例如月度事件池可具有如下结构：

| Month     | 代表性候选事件                                                   |
| --------- | --------------------------------------------------------- |
| January   | New Year gathering、家庭整理                                   |
| February  | Valentine dinner、Super Bowl gathering                     |
| March     | spring cleaning                                           |
| April     | spring gathering、Easter-related event                     |
| May       | Memorial Day gathering、graduation visit                   |
| June      | summer preparation、vacation preparation                   |
| July      | Independence Day、BBQ、summer gathering                     |
| August    | back-to-school preparation                                |
| September | Labor Day gathering、routine transition                    |
| October   | Halloween、decorations、party                               |
| November  | Thanksgiving、guest visit、大规模采购                            |
| December  | holiday decorations、gift/package arrival、family gathering |

这里所有事件都仅仅是 **candidate events**。

最终是否发生还取决于家庭 profile。

因此，不会出现每个 household 在 October 31 都完全相同的情况。

---

# 8. Hierarchical Behavior Generation

整个一个月的行为生成采用：

[
\boxed{
Month
\rightarrow
Week
\rightarrow
Day
\rightarrow
Activity
}
]

的层级结构。

参考的层级活动建模中，先建立 coarse general plan，再逐步分解成细粒度 indoor activity 能够更好地保持人物行为一致性和活动粒度。

---

## 8.1 Monthly Household Calendar

首先生成一个月级的 household calendar：

[
M_H=
{
E^{planned},
E^{periodic}
}.
]

这里不需要生成分钟级活动，只确定长期事件。

例如一个 31 天家庭可能形成：

| Day    | Long-term event       |
| ------ | --------------------- |
| Day 3  | grocery shopping      |
| Day 6  | weekly cleaning       |
| Day 10 | package delivery      |
| Day 13 | friends visit         |
| Day 17 | grocery shopping      |
| Day 21 | lamp failure          |
| Day 23 | buy replacement lamp  |
| Day 26 | birthday dinner       |
| Day 30 | monthly deep cleaning |

Monthly Calendar 是后续每日 schedule 的高层约束。

---

# 9. Weekly Routine Skeleton

对于每名居民 (u_i)，生成稳定 weekly template：

[
R_i=
{
R_i^{Mon},
R_i^{Tue},
\ldots,
R_i^{Sun}
}.
]

例如工作日：

[
07{:}00~WakeUp
]

[
07{:}20~Breakfast
]

[
08{:}00~LeaveHome
]

[
18{:}30~ReturnHome
]

[
19{:}00~Dinner
]

[
20{:}00~Leisure
]

[
23{:}00~Sleep.
]

Saturday 可能对应：

[
Breakfast
\rightarrow
GroceryShopping
\rightarrow
Leisure.
]

Sunday：

[
Breakfast
\rightarrow
HouseCleaning
\rightarrow
FamilyActivity.
]

这种 weekly skeleton 在一个月内重复，使：

[
Day~1,
Day~8,
Day~15,
Day~22
]

之间出现显著关联。

---

# 10. Daily Routine Sampling

真实居民不会每天精确在 7:20 早餐。

因此每一天的实际 activity time 从稳定 routine 周围采样：

[
t_a^{(d)}
\sim
\mathcal N(
\mu_a,
\sigma_a^2
).
]

例如：

[
\mu_{\text{breakfast}}=7{:}20.
]

实际一个月中可能出现：

[
07{:}12,\quad
07{:}27,\quad
07{:}18,\quad
07{:}34.
]

因此数据具有：

[
\text{regularity}
+
\text{variation}.
]

每日最终计划可以表示为

[
D_i^d
=====

F(
R_i^{weekday(d)},
E_d,
C_d,
X_{d,0},
\epsilon_i^d
).
]

其中：

[
\epsilon_i^d
]

表示当天的随机扰动。

---

# 11. Fine-Grained Indoor Activity Decomposition

Daily plan 中较粗粒度活动进一步展开成可以影响物体的 atomic activities。

例如：

[
Breakfast
]

可以展开为：

[
OpenRefrigerator
]

[
TakeApple
]

[
TakePlate
]

[
PrepareFood
]

[
EatBreakfast
]

[
PlacePlate
]

[
PlaceMug
]

[
WashDishes.
]

只有这些 fine-grained activities 才真正进入 environment transition module。

因此：

[
\text{Semantic Schedule}
\rightarrow
\text{Object-interaction Activities}.
]

---

# 12. Routine Events and Special Events

一个月的事件集合分成：

[
\mathcal E=
\mathcal E^{routine}
\cup
\mathcal E^{special}.
]

Routine event 包括高频行为，如早餐、洗澡、工作、看电视、做饭、洗碗和睡觉。

Special event 则进一步包含：

[
\mathcal E^{special}
====================

\mathcal E^{calendar}
\cup
\mathcal E^{social}
\cup
\mathcal E^{inventory}
\cup
\mathcal E^{maintenance}
\cup
\mathcal E^{personal}
\cup
\mathcal E^{stochastic}.
]

例如 Calendar Event 可以是 Thanksgiving，Social Event 可以是朋友来访，Inventory Event 可以是 grocery shopping，Maintenance Event 可以是 lamp failure，Personal Event 可以是 birthday，Stochastic Event 可以是临时聚餐。

---

# 13. Stateful Event Triggering

特殊事件不应该完全由随机概率产生，而要受到当前环境状态影响。

定义：

[
P(
e_d
\mid
C_d,
\Theta_H,
X_d,
E_{<d}
).
]

因此 event generator 同时考虑：

[
CalendarPrior,
HouseholdPreference,
CurrentState,
HistoricalEvents.
]

例如 grocery shopping 一方面具有固定周期：

[
Saturday
\rightarrow
P(Shopping)\uparrow,
]

另一方面也由库存触发：

[
Q_{\text{Apple}}<\tau_{\text{Apple}}
\rightarrow
P(Shopping)\uparrow.
]

因此购物既具有 habitual component，又具有 state-driven component。

---

# 14. Multi-Stage Special Events

重要事件不能被建模为一个瞬间。

将特殊事件表示为：

[
E^{special}
===========

E^{pre}
\rightarrow
E^{main}
\rightarrow
E^{post}.
]

例如家庭聚会。

### Preparation

居民提前购买：

[
Food,\ Drinks,\ Snacks,
DisposableCups.
]

因此：

[
Q_{\mathrm{Food}}\uparrow.
]

部分装饰物从 absent 变为 present。

### Main Event

聚会过程中：

[
Plate:
Cabinet
\rightarrow
DiningTable
]

[
Drink:
Refrigerator
\rightarrow
CoffeeTable
]

[
Snack:
Pantry
\rightarrow
CoffeeTable.
]

同时：

[
Q_{\mathrm{Food}}\downarrow.
]

### Aftermath

聚会结束之后：

[
DirtyDish\uparrow,
\qquad
Trash\uparrow.
]

随后 Cleaning Event：

[
DirtyDish
\rightarrow
Sink
\rightarrow
Cabinet
]

[
Trash
\rightarrow
Removed.
]

这样可以形成明显的 event-caused environmental pattern。

---

# 15. Event Dependency Graph

一个月内的重要事件之间具有依赖关系。

因此构建：

[
G_E=(V_E,D_E).
]

其中：

[
e_i\rightarrow e_j
]

表示 (e_j) 由 (e_i) 触发或依赖。

例如：

[
Consumption
\rightarrow
LowInventory
\rightarrow
Shopping
\rightarrow
Replenishment.
]

或者：

[
ObjectUsage
\rightarrow
Failure
\rightarrow
Removal
\rightarrow
Purchase
\rightarrow
Replacement.
]

因此整个模拟不是独立事件集合，而是具有因果链的长期过程。

---

# 16. Activity-Event Transition Model

为了将人类行为映射为环境变化，为每一个 activity/event (a) 建立一个结构化 transition model。

首先建模 activity location：

[
P(l\mid a,u,C_t).
]

然后确定 activity 涉及哪些物体：

[
P(o\mid a,l,u,X_t).
]

对于需要重新放置的物体：

[
P(
r
\mid
o,a,l,u,\Theta_H
).
]

进一步增加 effect type：

[
P(
\eta
\mid
o,a,X_t
),
]

其中

[
\eta\in
{
MOVE,
CONSUME,
ADD,
REMOVE,
DAMAGE,
REPAIR,
REPLACE
}.
]

因此：

[
Activity
\rightarrow
Location
\rightarrow
Objects
\rightarrow
Effects
\rightarrow
NewState.
]

---

# 17. Personalized Placement Distribution

为了让机器人学习到 resident-specific object distribution，每个居民与常用物体之间建立稳定 placement habit。

例如：

[
\pi_{u,o}(r)
============

P(
r
\mid
u,o
).
]

假设 Resident A 使用 Mug 后：

[
P(Desk)=0.55
]

[
P(KitchenCounter)=0.30
]

[
P(DiningTable)=0.15.
]

而 Resident B：

[
P(DiningTable)=0.50
]

[
P(CoffeeTable)=0.35
]

[
P(KitchenCounter)=0.15.
]

最终实际 placement probability 可以由多种因素融合：

[
P_t(r)
\propto
\alpha P_{\mathrm{habit}}(r)
+
\beta P_{\mathrm{activity}}(r)
+
\gamma P_{\mathrm{event}}(r)
+
\delta P_{\mathrm{noise}}(r).
]

其中

[
\alpha+\beta+\gamma+\delta=1.
]

通常：

[
\alpha,\beta>\delta.
]

因此物体存在规律，但不是确定性的。

---

# 18. Consumable Object Dynamics

对于食品、饮料、纸巾等 consumable objects，显式维护库存：

[
Q_c(t).
]

消费事件执行：

[
Q_c(t+\Delta t)
===============

\max
(
0,
Q_c(t)-k
).
]

例如：

[
Apple:
6\rightarrow5.
]

多人同时消费时：

[
Q_c(t+\Delta t)
===============

\max
\left(
0,
Q_c(t)
------

\sum_{u_i}k_i
\right).
]

当：

[
Q_c<\tau_c
]

时，购物概率上升。

购物之后：

[
Q_c(t^+)
========

Q_c(t^-)+K_c.
]

因此长期库存形成典型 saw-tooth pattern：

[
8\rightarrow6\rightarrow4
\rightarrow2\rightarrow1
\rightarrow9.
]

这类变化非常适合机器人学习 longer-term recurrence。

---

# 19. Object Introduction

一个月中允许：

[
\mathcal O_t\neq\mathcal O_{t+\Delta t}.
]

如果 Day 1 不存在 CoffeeMachine：

[
b_{\text{CoffeeMachine}}=0.
]

Day 12 发生：

[
Purchase(CoffeeMachine).
]

随后：

[
b_{\text{CoffeeMachine}}:
0\rightarrow1.
]

接下来新的物体还可以改变后续行为概率：

[
P(
MakeCoffee
\mid
CoffeeMachineExists
)

>

P(
MakeCoffee
\mid
CoffeeMachineAbsent
).
]

因此 special event 可以改变未来整个 activity distribution。

---

# 20. Object Damage, Removal and Replacement

耐用物体维护 condition：

[
h_o(t)\in[0,1].
]

随着长期使用：

[
h_o(t+\Delta t)
===============

h_o(t)-\Delta h.
]

并定义：

[
P(
Failure_o
\mid
h_o,
Usage_o
).
]

通常：

[
h_o\downarrow
\Rightarrow
P(Failure_o)\uparrow.
]

例如：

[
Lamp_A:
Normal
\rightarrow
Broken
\rightarrow
Removed.
]

随后：

[
Lamp_B:
Absent
\rightarrow
Purchased
\rightarrow
Installed.
]

从机器人的视角来看：

Day 5：

[
Lamp_A@Bedroom.
]

Day 17：

[
Lamp_A@Bedroom,\ Broken.
]

Day 20：

[
Lamp_A=Absent.
]

Day 24：

[
Lamp_B@Bedroom.
]

这构成真正的长期场景变化。

---

# 21. Temporary Object Dynamics

还有一类对象只在某一段时间出现。

例如：

[
Package,\ GiftBox,\ PartyDecoration,
GuestBag.
]

它们具有：

[
t_{\mathrm{enter}}
]

和

[
t_{\mathrm{leave}}.
]

因此：

[
b_o(t)=
\begin{cases}
0,&t<t_{\mathrm{enter}}\
1,&t_{\mathrm{enter}}\le t<t_{\mathrm{leave}}\
0,&t\ge t_{\mathrm{leave}}.
\end{cases}
]

这可以显著提高长期环境中的 novelty。

---

# 22. Formal Event Representation

最终每个事件统一表示为：

[
e_k=
(
id_k,
type_k,
U_k,
t_k^s,
t_k^e,
l_k,
Pre_k,
\Delta_k,
Parent_k
).
]

其中：

[
U_k
]

表示参与居民；

[
Pre_k
]

表示事件前置条件；

[
\Delta_k
]

表示对场景产生的变化。

例如：

[
e=
\text{Eat Apple}
]

具有前置条件：

[
Q_{\text{Apple}}>0.
]

effect 为：

[
CONSUME(Apple_i).
]

如果：

[
Q_{\text{Apple}}=0,
]

则该 activity 不允许直接执行。

系统必须改为 alternative food、skip 或触发 purchasing demand。

这保证了一个月状态的逻辑连续性。

---

# 23. Chronological State Transition

所有居民生成的 activities 和 household events 按真实时间排序：

[
E=
{
e_1,e_2,\ldots,e_N
},
]

满足：

[
t(e_1)\le
t(e_2)\le\cdots\le
t(e_N).
]

环境依次更新：

[
X_{k+1}
=======

T(
X_k,e_k
).
]

因此：

[
X_{Day~20}
]

天然包含过去 19 天发生的所有持久变化。

这是整个长期模拟器最核心的机制。

---

# 24. Concurrent Multi-Resident Events

多人家庭中存在活动重叠。

例如：

Resident A：

[
19{:}00~CookDinner
]

Resident B：

[
19{:}05~TakeDrink.
]

这种情况允许同时发生。

但是如果两个人同时需要唯一共享物体，例如：

[
TVRemote,
]

系统需要进行 availability check。

如果：

[
state(TVRemote)=InUseBy(u_1),
]

那么 (u_2) 的事件可以：

[
Delay,
Resample,
Skip
]

或者使用 alternative object。

因此不会产生同一物体同时存在于两个位置的问题。

---

# 25. Scene Grounding

High-level event generation只产生：

[
Room,
Object,
Receptacle
]

级别的语义关系。

真正应用到具体 house 时，再执行 scene grounding。

参考的场景配置方式也是先从活动推断 room、involved object 和 receptacle，再根据当前场景存在的 receptacle 对概率归一化并采样具体 object pose。

对于目标房间 (l)，只允许：

[
l\in\mathcal L_H.
]

对于目标 receptacle：

[
r\in\mathcal R(l).
]

如果概率模型给出：

[
P(r_1)=0.5,
P(r_2)=0.3,
P(r_3)=0.2,
]

但具体场景中不存在 (r_3)，则重新归一化：

[
P'(r_1)
=======

\frac{0.5}{0.8},
]

[
P'(r_2)
=======

\frac{0.3}{0.8}.
]

---

# 26. Physical Placement

选择 receptacle 后，在其有效 support surface 或 container volume 上生成具体 3D pose：

[
p_o
\sim
P(
p
\mid
r,o
).
]

Placement 需要满足：

[
CollisionFree(p_o)=True,
]

[
InsideValidRegion(p_o,r)=True,
]

以及合理的 object-receptacle relation。

例如：

[
Plate\rightarrow DiningTable
]

允许；

[
Plate\rightarrow Bed
]

在特定行为下可能允许；

而明显不合理的关系被约束。

对于 refrigerator、cabinet、drawer：

[
relation(o,r)=inside.
]

对于 table、desk：

[
relation(o,r)=on.
]

整个 semantic-to-3D grounding 与具体 house layout 解耦，使同一套居民行为模型能够应用到多个家庭场景。

---

# 27. Pattern-Preserving Stochasticity

整个数据生成的一个重要设计原则是：

[
\boxed{
Dynamics
========

Routine
+
Variation
+
LongTermChange
}
]

如果完全随机：

[
P(X_t|X_{<t})
\approx P(X_t),
]

机器人无法利用历史。

如果完全确定：

[
X_{t+7d}=X_t,
]

问题又过于简单。

因此通过 stochasticity 参数：

[
\epsilon
]

控制随机行为。

例如：

[
P(r)
====

(1-\epsilon)
P_{\mathrm{habit}}(r)
+
\epsilon
P_{\mathrm{random}}(r).
]

当：

[
\epsilon\rightarrow0,
]

场景规律非常稳定。

当：

[
\epsilon\rightarrow1,
]

环境趋近随机。

正常数据集需要使：

[
0<\epsilon<1.
]

这样机器人经过多次探索可以降低位置预测的不确定性，但永远不能简单记忆一个固定坐标。

---

# 28. Long-Term Distribution Shift

除了随机 variation，还主动产生 structural change。

例如前 15 天：

[
P(
Mug@Desk
)=0.15.
]

Day 16 居民开始长期在家办公后：

[
P(
Mug@Desk
)=0.55.
]

或者新增 CoffeeMachine 后：

[
P(
Mug@KitchenCounter
)
\uparrow.
]

因此：

[
P(X_t|\Theta)
]

本身允许随长期事件发生局部漂移。

这使机器人不仅需要：

[
Learn
]

还需要：

[
Update.
]

---

# 29. Household State Feedback

场景不是由 schedule 单向控制。

而采用：

[
Human
\leftrightarrow
Environment
]

双向过程。

Human activity 改变 environment：

[
Activity_t
\rightarrow
X_{t+1}.
]

但 environment 也反过来影响 future activity：

[
X_{t+1}
\rightarrow
Activity_{t+1}.
]

例如：

[
Apple=0
]

会降低 EatApple 的概率并提高：

[
Shopping
]

或 alternative food 的概率。

CoffeeMachine 被购买后提高：

[
MakeCoffee.
]

TV 损坏后降低：

[
WatchTV.
]

因此：

[
P(
A_t
\mid
X_t,\Theta,C_t
)
]

与：

[
P(
X_{t+1}
\mid
A_t,X_t
)
]

共同构成 closed-loop household simulation。

这个闭环对于一个月尺度尤其重要。

---

# 30. Scene Snapshot Generation

整个模拟内部以 event-driven 形式运行，但最终数据集需要提供机器人可以直接加载的 scene snapshots。

定义：

[
S(t)=
{
x_o(t)
}_{o\in\mathcal O}.
]

对于任意查询时间：

[
t_q,
]

均可以恢复：

[
S_{t_q}.
]

实际数据集建议同时保存两种 timestamp。

第一类是 regular snapshots，例如每天：

[
07{:}00,
12{:}00,
18{:}00,
22{:}00.
]

第二类是 event-aligned snapshots：

[
t_e^-,
\qquad
t_e^+.
]

也就是重要事件发生前后。

最终：

[
T_{\mathrm{save}}
=================

T_{\mathrm{periodic}}
\cup
T_{\mathrm{event}}.
]

因此一天内的动态和一个月尺度的变化都能够被保留下来。

---

# 31. Snapshot Content

每个 snapshot 包含当前所有 visible/present objects 的 ground-truth state：

[
S_t=
\left{
\begin{array}{l}
ObjectID,\
Category,\
Existence,\
Room,\
Receptacle,\
Position,\
Rotation,\
Condition,\
Owner,\
LastChangeTime
\end{array}
\right}.
]

另外保存：

[
Q_c(t)
]

用于表示 consumable inventory。

但对于机器人本身，应区分：

[
\text{Environment Observation}
]

和

[
\text{Generator Ground Truth}.
]

机器人只能通过视觉和自己的 exploration 获得物体信息。

诸如：

[
EventCause,
Owner,
FutureSchedule,
HouseholdProfile
]

可以保存在 ground truth 中，但不能直接暴露给 agent。

---

# 32. Data Organization

最终每个 household scene 在概念上包含以下数据层：

| Data layer           | 主要内容                          | Agent 是否直接可见 |
| -------------------- | ----------------------------- | ------------ |
| House Layout         | rooms、furniture、receptacles   | 是            |
| Resident Profiles    | residents、occupation、habit    | 否            |
| Household Profile    | shopping、tidiness、preferences | 否            |
| Monthly Calendar     | long-term events              | 否            |
| Resident Schedule    | minute-level activities       | 否            |
| Event Log            | 完整 causal events              | 否            |
| Object Lifecycle     | introduction/removal/failure  | 否            |
| Object State History | 完整 ground truth               | 否            |
| Scene Snapshots      | simulator state               | 环境加载         |
| Robot Observation    | RGB-D / semantic observations | 是            |

这样可以保证 benchmark 的核心问题仍然是：

> **机器人通过长期观察自己发现规律，而不是直接读取生成器产生的规律。**

---

# 33. Long-Term Consistency Verification

一个月的数据生成完成后，需要进行自动 consistency verification。

其中最重要的是 Temporal Consistency。

物体只有当前位置唯一：

[
Location(o,t)=1.
]

已经消耗的实例：

[
b_o=0
]

不能再次出现，除非 ADD 创建新的 instance。

状态还必须满足：

[
Consume(o)
\Rightarrow
b_o(t^-)=1.
]

以及：

[
Move(o)
\Rightarrow
b_o(t^-)=1.
]

购物引入物体：

[
ADD(o)
\Rightarrow
b_o(t^-)=0,
\quad
b_o(t^+)=1.
]

Replacement 必须满足：

[
REMOVE(o_{old})
+
ADD(o_{new}).
]

---

# 34. Causal Consistency

事件同样需要检查原因和结果。

例如不允许：

[
AppleInventory=0
]

但连续三天仍然 EatApple。

也不允许：

[
Lamp_A=Removed
]

之后居民再次 Move(Lamp_A)。

如果出现无效活动，应根据上下文执行：

[
Resample,
Alternative,
Delay,
Cancel.
]

而不是强制执行。

---

# 35. Spatial Consistency

每次 placement 都必须满足：

[
ValidRoom(o)
]

[
ValidReceptacle(o,r)
]

[
CollisionFree(o)
]

[
StablePlacement(o,r).
]

这保证最终 snapshot 真正能够在 embodied simulator 中加载。

---

# 36. Household Behavioral Consistency

还需要检查居民自身规律。

例如居民工作日通常 08:00 离家，则其 10:00 不应该频繁出现在家中做大量活动，除非：

[
WorkFromHome,
Holiday,
SickDay,
SpecialEvent
]

等 context 明确改变 schedule。

同样，如果 Resident A 有固定 bedroom：

[
Bedroom_A,
]

其私人 nighttime routine 通常应优先发生在那里。

---

# 37. Controllable Dynamics Parameters

整个生成器可以通过几个高层参数控制 difficulty：

| Parameter               | 控制内容                  |
| ----------------------- | --------------------- |
| (\epsilon_{routine})    | routine 时间随机程度        |
| (\epsilon_{placement})  | 物体位置随机程度              |
| (\lambda_{special})     | 特殊事件频率                |
| (\lambda_{intro})       | 新物体出现频率               |
| (\lambda_{failure})     | 物品损坏频率                |
| (\lambda_{consumption}) | consumable 消耗速度       |
| (\lambda_{visitor})     | 临时访客事件                |
| (\lambda_{drift})       | 长期 distribution shift |

这样同一个生成框架可以产生规律较强和规律较弱的家庭。

---

# 38. LLM 与结构化模拟器的职责划分

整个系统适合采用 **LLM semantic generation + structured state simulator** 的混合方案。

LLM 更适合负责：

| LLM component         | 功能                        |
| --------------------- | ------------------------- |
| Resident Generation   | 人物 profile                |
| Routine Planning      | weekly routine            |
| Monthly Planning      | 合理特殊事件                    |
| Activity Expansion    | high-level → fine-grained |
| Semantic Interaction  | activity 涉及哪些对象           |
| Event Effect Proposal | 特殊事件可能改变哪些对象              |

但 LLM **不直接控制最终环境状态**。

所有真正的 object state transition 都由 deterministic/probabilistic state engine 完成：

[
X_{t+1}=T(X_t,e_t).
]

因此可以避免 LLM 在 Day 20 忘记 Day 10 已经发生的事情。

这也是一个月生成任务中非常重要的架构设计。

---

# 39. 完整生成过程

从整个项目角度，一次完整 household scene generation 可以概括为以下唯一主流程：

1. 输入一个静态住宅环境，解析 room、receptacle 和初始 object inventory；根据 bedroom 数量和家庭配置生成若干长期居民，并初始化 resident profile 与 household latent profile。
2. 设置地理位置、月份和具体日期范围，建立一个月的 calendar context，并生成 calendar、personal、social 和周期性 monthly events。
3. 为每个居民生成稳定 weekly routine skeleton，并在整个模拟期间复用，而不是每天重新独立生成。
4. 对每一天根据 weekday、routine、特殊事件和当前 household state 生成 daily schedule，并加入有限时间扰动。
5. 将 high-level schedule 分解为能够实际作用于物体的 fine-grained activities。
6. 合并所有居民时间线，并插入 household-level joint events 和 state-triggered events。
7. 根据 Activity-Event Transition Model 确定 activity location、involved objects、target receptacles 和 transition type。
8. 按时间顺序执行全部事件，通过 MOVE、CONSUME、ADD、REMOVE、DAMAGE、REPAIR 和 REPLACE 更新 environment state。
9. 在每一步检查 object availability、inventory、resident ownership、event prerequisites 和 event dependencies。
10. 将语义 room-receptacle relations ground 到具体场景中的 3D coordinates，并执行碰撞、支撑关系和空间合法性检查。
11. 持续传播环境状态直到月底，同时允许 consumption、replenishment、introduction、failure 和 special events 改变后续行为分布。
12. 在 regular timestamps 和 important event timestamps 导出完整 scene snapshot，并保存 event log、object lifecycle 和 ground-truth state history。
13. 最后执行 temporal、causal、spatial 和 household consistency validation，得到可直接供 embodied robot 多次探索的一个月长期动态家庭数据。

---

# 40. 最终研究对象

这样生成的数据本质上不是：

[
\text{30 independent scenes}.
]

而是一条完整的：

[
\boxed{
\text{Household Evolution Trajectory}
}
]

即：

[
\mathcal T_H=
{
S_{t_0},
e_1,
S_{t_1},
e_2,
\ldots,
e_N,
S_{t_N}
}.
]

其中连续时间上的 snapshot 来自同一个 persistent household。

因此：

[
S_{Day~20}
]

和

[
S_{Day~1}
]

不是两个随机生成的房间，而是同一个家庭经过 19 天真实活动之后形成的两个状态。

最终希望机器人通过：

[
Observation_1,
Observation_2,
\ldots,
Observation_K
]

逐渐学习：

[
P(
Location(o)
\mid
Time,
History,
Context
),
]

进一步学习：

[
P(
Existence(o),
Location(o),
Quantity(o)
\mid
Time,
History
).
]

当长期事件发生后，还需要根据新 observation 对原来的长期规律进行更新。

因此整个项目的方法论核心可以最终归纳为：

[
\boxed{
\textbf{
Persistent Household
+
Hierarchical Routine
+
Event-driven Evolution
+
Object Lifecycle
+
Causal State Propagation
}}
]

它们共同形成一个**具有日内周期、周级重复、月级事件、物体生命周期和长期非平稳性的家庭场景生成模型**。这样生成的数据尤其适合研究 lifelong object memory、spatio-temporal object prediction、long-term ObjectNav，以及机器人在重复访问同一环境时的经验积累与记忆更新。

---

# 当前工程实现注记（2026-08-13）

本文是 Lifespan 方法论草案，描述的是最终目标。当前代码已经实现第一版 semantic-only MVP，对应方法论中的高层家庭建模、月/日行为建模、事件展开和状态传播部分：

```text
core/lifespan_generate_layouts.py
core/lifespan_household_generator.py
core/lifespan_profiles.py
core/lifespan_event_generator.py
core/lifespan_state_engine.py
```

当前已能生成：

```text
household_profile.json
household_relationship_graph.json
resident_daily_routines.json
daily_important_events.json
object_lifespan_profiles.json
event_log.json
state_history.json
snapshot_requests.json
manifest.json
layouts/snapshot_*.json
```

当前尚未完成方法论中的 Scene Grounding and Physical Validation 闭环。也就是说，`layouts/snapshot_*.json` 仍是 semantic-only layout，物体 `position/rotation` 暂为 `null`，需要后续接入 `assign_objects_to_receptacle_instances.py` 和 `place_objects_on_instances.py` 才能成为最终可加载的 Habitat 3D layout。
