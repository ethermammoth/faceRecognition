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
    
    firstName = "";
    lastName = "";
    
    setupgui();
    
    imgLocationBuffer = ofBufferFromFile("train.txt");
    if(imgLocationBuffer.size()){
        while(!imgLocationBuffer.isLastLine())
        {
            string line = imgLocationBuffer.getNextLine();
            if(!line.empty())
                imgLocation.push_back(line);
        }
    }
    
#ifdef USE_UVC_CONTROLS
    uvcControl.useCamera(0x5ac, 0x8507, 0x00);
#endif
    
}

//--------------------------------------------------------------
void faceRecognition::update(){
    //update EVM parameters
    evm.setTemporalFilter(filter);
    evm.setParamsIIR(alpha_iir, lambda_c_iir, r1, r2, chromAttenuation_iir);
    evm.setParamsIdeal(alpha_ideal, lambda_c_ideal, fl, fh, samplingRate, chromAttenuation_ideal);
    
    
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
            
            evm.update(toCv(faceImage));
        }
    }
}

//--------------------------------------------------------------
void faceRecognition::draw(){
    
    ofBackgroundGradient(ofColor(64), ofColor(0));
    ofSetColor(255, 255, 255);
    
    ofDrawBitmapStringHighlight("CAM STREAM", 20.f, 40.f);
    ofSetColor(255, 0, 0);
    ofRect(10.f, 50.f, camWidth + 20.f, camHeight + 20.f);
    ofSetColor(255, 255, 255);
    cam.draw(20.f, 60.f);
    
    ofDrawBitmapStringHighlight("EXTRACTED FACE", camWidth + 50, 40);
    ofSetColor(255, 0, 0);
    ofRect(camWidth + 50, 50, 170, 170);
    
    ofSetColor(255, 255, 255);
    ofDrawBitmapStringHighlight("EQUALIZED FACE", camWidth + 50, 240);
    ofSetColor(255, 0, 0);
    ofRect(camWidth + 50, 250, 170, 170);
    
    ofSetColor(255, 255, 255);
    ofDrawBitmapStringHighlight("MOTION AMPLIFIED", camWidth + 50, 440);
    ofSetColor(255, 0, 0);
    ofRect(camWidth + 50, 450, 170, 170);
    
    if(faceFound) {
        ofSetColor(255, 255, 255);
        faceCvColor.draw(camWidth + 60, 60);
        faceCvGray.draw(camWidth + 60, 260);
        evm.draw(camWidth + 60, 460);
        ofPushMatrix();
        ofTranslate(20.f, 60.f);
        camTracker.draw();
        ofPopMatrix();
    }
}

//--------------------------------------------------------------
void faceRecognition::saveFaceImage(ofxCvGrayscaleImage img){
    ofImage saveImg;
    saveImg.setFromPixels(img.getPixels(), img.width, img.height, OF_IMAGE_GRAYSCALE);
    
    string imgName = "faces/";
    imgName += uiFirstName->getTextString() + uiLastName->getTextString();
    imgName += ".jpg";
    //imgLocationBuffer.append(imgName);
    //PATH FACES _FirstName_LastName . fileFormat
    //Evtl noch XML datei / bzw txt datei mit mehr daten
    
    cout << "saving to: " << imgName << endl;
    
    saveImg.saveImage(imgName);
}

//--------------------------------------------------------------
void faceRecognition::guiEvent(ofxUIEventArgs &e){
    string name = e.widget->getName();
	int kind = e.widget->getKind();
    
    if(name == "FirstName"){
        firstName = uiFirstName->getTextString();
    }
    
    if(name == "LastName"){
        lastName = uiLastName->getTextString();
    }
    
    if(name == "Temporal IIR"){
        filter = EVM_TEMPORAL_IIR;
    }
    
    if (name == "Temporal Ideal (Unimplemented)"){
        filter = EVM_TEMPORAL_IDEAL;
    }
    
    if(name == "save"){
        if(uiFirstName->getTextString().length() > 0 &&
           uiLastName->getTextString().length() > 0)
        {
            saveFaceImage(faceCvGray);
        }else{
            uiFeedback->setLabel("enter your name first!");
        }
    }
    
    cout << "UI event: " << name << endl;
}

//--------------------------------------------------------------
void faceRecognition::keyPressed(int key){

}

//--------------------------------------------------------------
void faceRecognition::keyReleased(int key){
    if(key == OF_KEY_DOWN){
        saveFaceImage(faceCvGray);
    }
#ifdef USE_UVC_CONTROLS    
    if(key == OF_KEY_UP){
        uvcControl.setAutoExposure(!uvcControl.getExposure());
        uvcControl.setAutoFocus(!uvcControl.getAutoFocus());
        uvcControl.setAutoWhiteBalance(!uvcControl.getAutoWhiteBalance());
        uvcExposure = uvcControl.getExposure();
    }
    
    if(key == OF_KEY_RIGHT){
        uvcExposure += 0.05;
        uvcExposure = ofClamp(uvcExposure, 0.f, 1.f);
    }
    
    if(key == OF_KEY_LEFT){
        uvcExposure -= 0.05;
        uvcExposure = ofClamp(uvcExposure, 0.f, 1.f);
    }
#endif
}

//--------------------------------------------------------------
void faceRecognition::setupgui(){
    //GUI SETUP
    uiCanvas = new ofxUICanvas(20.f, camHeight + 85.f, 250.f, 200.f);
    uiCanvas->setName("datainput");
    
    uiFirstName = new ofxUITextInput("FirstName", "First Name", 200.f);
    uiLastName = new ofxUITextInput("LastName", "Last Name", 200.f);
    uiFeedback = new ofxUILabel("feedback", "OK", 1);
    
    uiCanvas->addLabel("DATA INPUT", OFX_UI_FONT_MEDIUM);
    uiCanvas->addWidgetDown(uiFeedback);
    uiCanvas->addSpacer();
    uiCanvas->addWidgetDown(uiFirstName);
    uiCanvas->addWidgetDown(uiLastName);
    uiCanvas->addSpacer();
    uiCanvas->addButton("save", false);
    uiCanvas->addSpacer();
    
    ofAddListener(uiCanvas->newGUIEvent,this,&faceRecognition::guiEvent);
    
    //EVM GUI
    uiEvm = new ofxUICanvas(camWidth + 240.f, 25.f, 285.f, 400.f);
    uiEvm->setName("evm");
    
    float length = 280.f;
    float dim = 20.f;
    
    uiEvm->addLabel("EULERIAN VIDEO MAGNIFICATION", OFX_UI_FONT_MEDIUM);
    uiEvm->addWidgetDown(new ofxUIFPS(OFX_UI_FONT_MEDIUM));
    uiEvm->addSpacer(length, 2);
	uiEvm->addLabel("TEMPORAL FILTER", OFX_UI_FONT_MEDIUM);
    temporal_filter.push_back("Temporal IIR");
    temporal_filter.push_back("Temporal Ideal (Unimplemented)");
    uiEvm->addRadio("SELECT TEMPORAL FILTER TYPE", temporal_filter, OFX_UI_ORIENTATION_VERTICAL, dim, dim);
    
    uiEvm->addSpacer(length, 2);
    uiEvm->addLabel("IIR FILTER PARAMETER", OFX_UI_FONT_MEDIUM);
    uiEvm->addSlider("Amplification", 0, 100, &alpha_iir, length, dim);
    uiEvm->addSlider("Cut-off Wavelength", 0, 100, &lambda_c_iir, length, dim);
    uiEvm->addSlider("r1 (Low cut-off?)", 0, 1, &r1, length, dim);
    uiEvm->addSlider("r2 (High cut-off?)", 0, 1, &r2, length, dim);
	uiEvm->addSlider("ChromAttenuation", 0, 1, &chromAttenuation_iir, length, dim);
    
    uiEvm->addSpacer(length, 2);
    uiEvm->addLabel("IDEAL FILTER PARAMETER", OFX_UI_FONT_MEDIUM);
    uiEvm->addSlider("Amplification", 0, 200, &alpha_ideal, length, dim);
    uiEvm->addSlider("Cut-off Wavelength", 0, 100, &lambda_c_ideal, length, dim);
    uiEvm->addSlider("Low cut-off", 0, 10, &fl, length, dim);
    uiEvm->addSlider("High cut-off", 0, 10, &fh, length, dim);
    uiEvm->addSlider("SamplingRate", 1, 60, &samplingRate, length, dim);
    uiEvm->addSlider("ChromAttenuation", 0, 1, &chromAttenuation_ideal, length, dim);
    
    ofAddListener(uiEvm->newGUIEvent, this, &faceRecognition::guiEvent);
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

void faceRecognition::exit(){
    delete uiCanvas;
    delete uiEvm;
}