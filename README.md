
[<img src='./docs/genmos_demogrid.png' width='450px'>](https://www.youtube.com/watch?v=TfCe2ZVwypU)

(click image to watch video)

# GenMOS (Generalized Multi-Object Search)

To get started, check out [this wiki page](https://github.com/zkytony/genmos_object_search/wiki/100-GenMOS:-A-System-for-Generalized-Multi-Object-Search).

This repo is structured as follows:
- [genmos_object_search](./genmos_object_search): a robot-independent package for object search that implements the GenMOS system.
- [ros](./ros): a ROS package that integrates `genmos_object_search` with ROS.
- [ros2](./ros2) a ROS 2 package that integrates `genmos_object_search` with ROS 2 (TODO)
- [viam](./viam): a Viam client application that integrates `genmos_object_search` with [Viam](https://www.viam.com/).

Check out the README in each directory for more information.

[Design.org](./Design.org) contains a system architecture diagram and a documentation of the design of GenMOS as it was developed.

**Demo video:** https://www.youtube.com/watch?v=TfCe2ZVwypU


## Paper Reference

[*A System for Generalized 3D Multi-Object Search*](https://kaiyuzheng.me/documents/papers/icra23-genmos.pdf).
Kaiyu Zheng, Anirudha Paul, Stefanie Tellex. International Conference on Robotics and Automation (ICRA) 2023
```bibtex
@inproceedings{zheng2023asystem,
  title={A System for Generalized 3D Multi-Object Search,
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  author={Zheng, Kaiyu and Paul, Anirudha and Tellex, Stefanie},
  year={2023}
}
```
Additionally, refer to Kaiyu's PhD Thesis [*Generalized Object Search*](https://kaiyuzheng.me/documents/papers/dissertation.pdf) (6.7MB) for (1) details on the additional deployment of GenMOS on the Universal Robotics UR5e arm using [Viam](https://www.viam.com/) as the middleware;  (2) a literature survey and taxonomy on object search; (3) and more...
