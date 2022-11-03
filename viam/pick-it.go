package main

import (
	"context"

	"github.com/edaniels/golog"
	"go.viam.com/rdk/config"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/robot/client"
	"go.viam.com/rdk/services/motion"
	"go.viam.com/rdk/services/vision"
	"go.viam.com/rdk/spatialmath"
	"go.viam.com/rdk/utils"
	"go.viam.com/utils/rpc"

	// registers all components.
	"go.viam.com/rdk/components/arm"
	_ "go.viam.com/rdk/components/register"
	// registers the vision service
	_ "go.viam.com/rdk/services/vision/register"
)

func main() {
	logger := golog.NewDevelopmentLogger("pick-it")
	r, err := client.New(
		context.Background(),
		"macbook.496koy7yd1.viam.cloud",
		logger,
		client.WithDialOptions(rpc.WithCredentials(rpc.Credentials{
			Type:    utils.CredentialsTypeRobotLocationSecret,
			Payload: "onq7ch977yafwxm7tuqlvfwuqlzmgu730mvn1exrr6w67t1s",
		})),
	)
	if err != nil {
		logger.Fatalf("cannot create robot: %v", err)
	}
	defer r.Close(context.Background())
	visionSrv, err := vision.FromRobot(r, "builtin")
	if err != nil {
		logger.Fatalf("cannot get vision service: %v", err)
	}
	// add detector
	err = visionSrv.AddDetector(
		context.Background(),
		vision.VisModelConfig{
			Name: "find_objects",
			Type: "tflite_detector",
			Parameters: config.AttributeMap{
				"model_path":  "/Users/bijanh/pick-it-bot/data/effDet0.tflite",
				"num_threads": 1,
			},
		},
	)
	if err != nil {
		logger.Fatalf("cannot add tflite model: %v", err)
	}
	// add 3D segmenter
	err = visionSrv.AddSegmenter(
		context.Background(),
		vision.VisModelConfig{
			Name: "find_objects_segmenter",
			Type: "detector_segmenter",
			Parameters: config.AttributeMap{
				"confidence_threshold_pct": 0.3,
				"detector_name":            "find_objects",
			},
		},
	)
	if err != nil {
		logger.Fatalf("cannot add segmenter: %v", err)
	}
	logger.Info("Resources:")
	logger.Info(r.ResourceNames())
	// find the coordinates of the object to move to (just grabs the first one, but you can filter by label)
	var myPose spatialmath.Pose
	for {
		objs, err := visionSrv.GetObjectPointClouds(context.Background(), "my-3D-camera-name", "find_objects_segmenter")
		if err != nil {
			logger.Errorf("cannot get segments: %v", err)
		}
		logger.Infof("number of objects: %d", len(objs))
		if len(objs) > 0 {
			myPose = objs[0].Geometry.Pose()
			break
		}
	}
	logger.Infof("Location of object: %v", myPose.Point())
	// get the motion service
	motionSrv, err := motion.FromRobot(r, "builtin")
	if err != nil {
		logger.Fatalf("cannot get motion service: %v", err)
	}
	moveToLocation := referenceframe.NewPoseInFrame("my-3D-camera-name", myPose)
	_, err = motionSrv.Move(context.Background(), arm.Named("my-arm-to-move-name"), moveToLocation, nil, nil)
	if err != nil {
		logger.Fatalf("cannot move: %v", err)
	}
}