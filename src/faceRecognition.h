#pragma once

#include "ofMain.h"
#include "ofxCv.h"
using namespace ofxCv;
using namespace cv;

#include "ofxFaceTrackerThreaded.h"

#include "ofxCvFaceRec.h"

class faceRecognition : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed  (int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

        ofVideoGrabber cam;
        ofxFaceTrackerThreaded camTracker;
    
        int camWidth, camHeight;
        int finalSize;
        bool faceFound;
    
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
};
