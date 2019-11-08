#include "main.h"

int main(int argc, char *argv[]) {
	std::string mainfolder = std::string(argv[1]);
	int startFrame = atoi(argv[2]);
	int endFrame = atoi(argv[3]);
	std::cout << mainfolder << " " << startFrame << " " << endFrame << std::endl;
	//test_BlenderDataAllPlanesweep();
	//test_FaroDataAllPlanesweep();
	//test_FaroDataAll();
	//test_VehicleSegmentationSequence();
	//test_VehicleSegmentation();
	//test_PlaneSweepWithTvl1();
	//test_ImageSequencePlanesweep();
	//test_PlaneSweep();
	test_ImageSequence(mainfolder, startFrame, endFrame);
	//test_ImageSequenceLite();
	//test_StereoLiteTwoFrames(4, 2.0f, 30, 30);
	//test_TwoFrames(4, 2.0f, 30, 30);
	//test_LimitingRangeOne();
	//test_BlenderDataSequence();
	//test_ImageSequence();
	/*for (int k = 1; k <= 20; k++) {
		test_Timing(k);
	}*/
	//test_LimitingRange();
	//test_FaroData();
	//test_BlenderData();
	//test_IcraAddedAccuratePixels();
	return 0;
}