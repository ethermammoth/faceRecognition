#pragma once

#include "ofMain.h"
#include "ofxCv.h"
using namespace ofxCv;
using namespace cv;

#include "ofxFaceTrackerThreaded.h"
#include "ofxCvFaceRec.h"

#include "ofxUI.h"

//Camera Settings OSX only
#include "ofxUVC.h"

#include "ofxEvm.h"

//#define USE_UVC_CONTROLS


class faceRecognition : public ofBaseApp{

public:
    void setup();
    void update();
    void draw();
    void exit();
    void setupgui();

    void keyPressed  (int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);

    void saveFaceImage(ofxCvGrayscaleImage img);
    void loadFaceImages();

    void guiEvent(ofxUIEventArgs &e);
    
    string getPersonName(int _id);
    
    ofVideoGrabber cam;
    ofxFaceTrackerThreaded camTracker;

    int camWidth, camHeight;
    int finalSize;
    bool faceFound;
    bool faceNewFound;
    int faceId;

    ofPolyline faceOutline;
    ofRectangle faceBB;
    ofPath faceSolid;
    ofFbo maskFbo, resultFbo;
    ofShader maskShader;

    ofImage faceImage;
    
    int smoothFactor;

    //CV
    ofxCvColorImage faceCvColor;
    ofxCvGrayscaleImage faceCvGray;
    //Main rec
    ofxCvFaceRec recognizer;
    
    //Pulse detection
    ofxEvm evm;
    vector<string> temporal_filter;
    EVM_TEMPORAL_FILTER filter = EVM_TEMPORAL_IIR;
    //iir params
    float alpha_iir = 10;
    float lambda_c_iir = 16;
    float r1 = 0.4;
    float r2 = 0.05;
    float chromAttenuation_iir = 0.1;
    //ideal params
    float alpha_ideal = 150;
    float lambda_c_ideal = 6;
    float fl = 140.0/60.0;
    float fh = 160.0/60.0;
    float samplingRate = 30;
    float chromAttenuation_ideal = 1;

#ifdef USE_UVC_CONTROLS
    //Canera settings
    //TODO: mac only
    ofxUVC uvcControl;
    float uvcExposure;
#endif
    
    //GUI
    ofxUICanvas *uiCanvas;
    ofxUITextInput *uiFirstName;
    ofxUITextInput *uiLastName;
    ofxUILabel *uiFeedback;
    ofxUIToggle *uiEnableEvm;
    ofxUIToggle *uiDoRecognize;
    string firstName, lastName;
    ofxUICanvas *uiEvm;
    bool evmEnable;
    bool doRecognize;
    
    //face db
    int person;
    int bestPerson;
    double recLeastSquareDist;
    vector<string> trainingImages;
    const string filename = "train.txt";
};
