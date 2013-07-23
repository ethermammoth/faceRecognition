//
//  ofxFaceTrackerResult.cpp
//  faceRecognition
//
//  Created by Rasmus on 7/23/13.
//
//

#include "ofxFaceTrackerResult.h"

ofxFaceTrackerResult::ofxFaceTrackerResult(){
    faceFound = false;
    person = -1;
    bestPerson = -1;
    recLeastSquareDist = std::numeric_limits<double>::max();
    faceNewFound = true;
    faceId = 0;
    initialized = false;
}

void ofxFaceTrackerResult::setup(int camWidth, int camHeight, int finalSize){
    maskFbo.allocate(camWidth, camHeight, GL_RGB);
    resultFbo.allocate(camWidth, camHeight, GL_RGB);
    faceImage.allocate(finalSize, finalSize, OF_IMAGE_COLOR);
    faceCvColor.allocate(finalSize, finalSize);
    faceCvGray.allocate(finalSize, finalSize);
    faceImage.setUseTexture(false);
    initialized = true;
}