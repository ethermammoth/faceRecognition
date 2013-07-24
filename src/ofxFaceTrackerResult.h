//
//  ofxFaceTrackerResult.h
//  faceRecognition
//
//  Created by Rasmus on 7/23/13.
//
//

#ifndef faceRecognition_ofxFaceTrackerResult_h
#define faceRecognition_ofxFaceTrackerResult_h
#include "ofMain.h"
#include "ofxOpenCv.h"

class ofxFaceTrackerResult{

public:
    
    ofxFaceTrackerResult();
    
    void setup(int camWidth, int camHeight, int finalSize);
    
    ofPolyline faceOutline;
    ofRectangle faceBB;
    ofPath faceSolid;
    ofFbo maskFbo, resultFbo;
    ofImage faceImage;
    //CV
    ofxCvColorImage faceCvColor;
    ofxCvGrayscaleImage faceCvGray;
    
    //Eye
    vector<ofColor> eyeColors;
    ofImage eyeImage;
    
    //Skin Color
    ofColor skinColor;
    
    bool faceFound;
    bool faceNewFound;
    int faceId;
    
    bool eyeFound;
    bool skinFound;
    
    //face db
    int person;
    int bestPerson;
    double recLeastSquareDist;
    
    bool initialized;
    
};




#endif
