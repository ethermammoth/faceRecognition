#include "faceRecognition.h"

//--------------------------------------------------------------
void faceRecognition::setup(){
    cam.initGrabber(640, 480);
    camTracker.setup();
}

//--------------------------------------------------------------
void faceRecognition::update(){
    cam.update();
    if(cam.isFrameNew()) {
        camTracker.update(toCv(cam));
    }
}

//--------------------------------------------------------------
void faceRecognition::draw(){
    cam.draw(0, 0);
    
    if(camTracker.getFound()) {
        camTracker.draw();
    }
    
    ofDrawBitmapStringHighlight("FPS: " + ofToString(ofGetFrameRate()), 15, 15);

}

//--------------------------------------------------------------
void faceRecognition::keyPressed(int key){

}

//--------------------------------------------------------------
void faceRecognition::keyReleased(int key){

}

//--------------------------------------------------------------
void faceRecognition::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void faceRecognition::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void faceRecognition::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void faceRecognition::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void faceRecognition::windowResized(int w, int h){

}

//--------------------------------------------------------------
void faceRecognition::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void faceRecognition::dragEvent(ofDragInfo dragInfo){ 

}