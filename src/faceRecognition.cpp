#include "faceRecognition.h"

//--------------------------------------------------------------
void faceRecognition::setup(){
    faceFound = false;
    cam.initGrabber(640, 480);
    camTracker.setup();
}

//--------------------------------------------------------------
void faceRecognition::update(){
    cam.update();
    if(cam.isFrameNew()) {
        
        /*
         * PART 1 - Face Detection and equalization
         * 1) Detect face with ofxFaceTracker
         * 2) Get the Face Img
         * 3) Convert to Grayscale and Histogram equalize
         */
        
        camTracker.update(toCv(cam));
        faceFound = camTracker.getFound();
        if(faceFound)
        {
            faceOutline = camTracker.getImageFeature(ofxFaceTracker::ALL_FEATURES);
            faceBB = faceOutline.getBoundingBox();
            //resize(image, im, cv::Size(rescale * image.cols, rescale * image.rows));
            //cvtColor(im, gray, CV_RGB2GRAY);
        }
    }
}

//--------------------------------------------------------------
void faceRecognition::draw(){
    ofBackground(0);
    ofSetColor(255, 255, 255);
    cam.draw(0, 0);
    
    ofSetColor(255, 0, 0);
    if(camTracker.getFound()) {
        ofRect(faceBB);
        ofSetColor(255, 255, 255);
        faceOutline.draw();
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