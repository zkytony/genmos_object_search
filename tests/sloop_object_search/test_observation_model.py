

### Unit test ###
def unittest():
    """THIS IS OUTDATED TEST"""
    from sloop_object_search.oopomdp.env.env import make_laser_sensor,\
        make_proximity_sensor, equip_sensors,\
        interpret, interpret_robot_id
    # Test within search region check,
    # and the observation model probability and
    # sampling functions.
    worldmap =\
        """
        ..........
        ....T.....
        ......x...
        ..T.r.T...
        ..x.......
        ....T.....
        ..........
        """
       #0123456789
       # 10 x 8
    worldstr = equip_sensors(worldmap,
                             {"r": make_laser_sensor(90, (1,5), 0.5, False)})
    env = interpret(worldstr)
    robot_id = interpret_robot_id("r")
    robot_pose = env.state.pose(robot_id)

    # within_range test
    sensor = env.sensors[robot_id]
    assert sensor.within_range(robot_pose, (4,3)) == False
    assert sensor.within_range(robot_pose, (5,3)) == True
    assert sensor.within_range(robot_pose, (6,3)) == True
    assert sensor.within_range(robot_pose, (7,2)) == True
    assert sensor.within_range(robot_pose, (7,3)) == True
    assert sensor.within_range(robot_pose, (4,3)) == False
    assert sensor.within_range(robot_pose, (2,4)) == False
    assert sensor.within_range(robot_pose, (4,1)) == False
    assert sensor.within_range(robot_pose, (4,5)) == False
    assert sensor.within_range(robot_pose, (0,0)) == False

    print(env.state)

    # observation model test
    O0 = ObjectObservationModel(0, sensor, (env.width, env.length), sigma=0.01, epsilon=1)
    O2 = ObjectObservationModel(2, sensor, (env.width, env.length), sigma=0.01, epsilon=1)
    O3 = ObjectObservationModel(3, sensor, (env.width, env.length), sigma=0.01, epsilon=1)
    O5 = ObjectObservationModel(5, sensor, (env.width, env.length), sigma=0.01, epsilon=1)

    z0 = O0.sample(env.state, Look)
    assert z0.pose == ObjectObservation.NULL
    z2 = O2.sample(env.state, Look)
    assert z2.pose == ObjectObservation.NULL
    z3 = O3.sample(env.state, Look)
    assert z3.pose == (6, 3)
    z5 = O5.sample(env.state, Look)
    assert z5.pose == ObjectObservation.NULL

    assert O0.probability(z0, env.state, Look) == 1.0
    assert O2.probability(z2, env.state, Look) == 1.0
    assert O3.probability(z3, env.state, Look) >= 1.0
    assert O3.probability(ObjectObservation(3, ObjectObservation.NULL),
                          env.state, Look) == 0.0
    assert O5.probability(z5, env.state, Look) == 1.0

if __name__ == "__main__":
    unittest()
