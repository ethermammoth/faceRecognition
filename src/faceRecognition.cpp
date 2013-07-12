#include "faceRecognition.h"

//--------------------------------------------------------------
void faceRecognition::setup(){
    
    ofEnableAlphaBlending();
    camWidth = 640;
    camHeight = 480;
    
    //Size of the Image used for recognition (Square)
    finalSize = 150;
    
    faceFound = false;
    cam.initGrabber(camWidth, camHeight);
    camTracker.setup();
    
    //------------------
    // Allocate Images
    //------------------
    maskFbo.allocate(camWidth, camHeight, GL_RGB);
    resultFbo.allocate(camWidth, camHeight, GL_RGB);
    faceImage.allocate(finalSize, finalSize, OF_IMAGE_COLOR);
    faceCvColor.allocate(finalSize, finalSize);
    faceCvGray.allocate(finalSize, finalSize);
    
    faceImage.setUseTexture(false);
    
    string shaderProgram = "#version 120\n \
    #extension GL_ARB_texture_rectangle : enable\n \
    \
    uniform sampler2DRect tex0;\
    uniform sampler2DRect maskTex;\
    \
    void main (void){\
    vec2 pos = gl_TexCoord[0].st;\
    \
    vec3 src = texture2DRect(tex0, pos).rgb;\
    float mask = texture2DRect(maskTex, pos).r;\
    \
    gl_FragColor = vec4( src , mask);\
    }";
    
    maskShader.setupShaderFromSource(GL_FRAGMENT_SHADER, shaderProgram);
    maskShader.linkProgram();
    
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
            
            //------------------
            // Here we start by masking the face out of the camera image
            //------------------
            faceOutline = camTracker.getImageFeature(ofxFaceTracker::FACE_OUTLINE);
            faceBB = faceOutline.getBoundingBox();
            faceSolid.clear();
            
            for(int i=0; i < faceOutline.getVertices().size(); i++) {
                if(i == 0) {
                    faceSolid.newSubPath();
                    faceSolid.moveTo(faceOutline.getVertices()[i]);
                }else{
                    faceSolid.lineTo(faceOutline.getVertices()[i]);
                }
            }
            
            faceSolid.close();
            
            maskFbo.begin();
            ofClear(0, 255);
            faceSolid.draw();
            maskFbo.end();
            
            resultFbo.begin();
            ofClear(0, 0, 0, 0);
            maskShader.begin();
            maskShader.setUniformTexture("maskTex", maskFbo.getTextureReference(), 1);
            cam.draw(0, 0);
            maskShader.end();
            resultFbo.end();
            
            //------------------
            // Resulting Image from FBO cropped and resized
            //------------------
            ofPixels pix;
            resultFbo.readToPixels(pix);            
            faceImage.setFromPixels(pix);
            faceImage.crop(faceBB.position.x, faceBB.position.y, faceBB.width, faceBB.height);
            faceImage.resize(finalSize, finalSize);
            
            faceCvColor.setFromPixels(faceImage.getPixels(), faceImage.getWidth(), faceImage.getHeight());
            faceCvGray = faceCvColor;
            //faceCvGray.contrastStretch();            
            cvEqualizeHist(faceCvGray.getCvImage(), faceCvGray.getCvImage());
            
        }
    }
}

//--------------------------------------------------------------
void faceRecognition::draw(){
    ofBackground(0);
    ofSetColor(255, 255, 255);
    cam.draw(0, 0);
    
    ofDrawBitmapStringHighlight("EXTRACTED FACE", camWidth + 50, 40);
    ofSetColor(255, 0, 0);
    ofRect(camWidth + 50, 50, 170, 170);
    
    ofSetColor(255, 255, 255);
    ofDrawBitmapStringHighlight("EQUALIZED FACE", camWidth + 50, 240);
    ofSetColor(255, 0, 0);
    ofRect(camWidth + 50, 250, 170, 170);
    
    if(camTracker.getFound()) {
        ofSetColor(255, 255, 255);
        faceCvColor.draw(camWidth + 60, 60);
        faceCvGray.draw(camWidth + 60, 260);
        camTracker.draw();
    }
    
    string feedbackTxt = "FPS: " + ofToString(ofGetFrameRate());
    ofDrawBitmapStringHighlight(feedbackTxt, 15, 15);

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