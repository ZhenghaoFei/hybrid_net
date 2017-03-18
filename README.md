## Planing - Reaction Hybrid Network

We are investigating a planning-reaction hybrid network for deep reinforcement learning robot control.
The basic idea is design a planning network and  combine it to a reaction network.

#### Reaction Network
We call the standard plain network as reaction network, which means they are good at reaction but not good at planning computaion.

A standard reaction network is like this:

![alt text](https://github.com/ZhenghaoFei/hybrid_net/blob/master/resources/plain.png "Reaction Network")



#### Planning Network
The idea of planning network was got from :

*Tamar, Aviv, et al. "Value iteration networks." Advances in Neural Information Processing Systems. 2016.*

We simplfied the planning network and make it more general do not rely on explicit value interation.

We design a planning network use rnn like this:
![alt text](https://github.com/ZhenghaoFei/hybrid_net/blob/master/resources/rnnplanning.png "Planning Network")


#### Planing - Reaction Hybrid Network
Our ultimate purpose is design a network that has Planing ability while also good at Reaction, the combining of two kinds of
computation can improve the total performance of deep reinforcement learning especially in control.

The preliminary design is like this:
![alt text](https://github.com/ZhenghaoFei/hybrid_net/blob/master/resources/hybrid.png "Reaction Network")

We are using a very small network "state evaluation net" in the middle and every time it will output a scala **alpha** and combine
the ouput of planning and reaction module by the ratio of alpha, 1 - alpha. We hope the "state evaluation net" can learn to know
when the situation is needing planning more.




