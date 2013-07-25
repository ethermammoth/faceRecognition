#include "faceRecognition.h"

//--------------------------------------------------------------
void faceRecognition::setup(){
    
    ofEnableAlphaBlending();
    ofSetFrameRate(60);
    ofSetVerticalSync(true);
    ofSetDrawBitmapMode(OF_BITMAPMODE_MODEL_BILLBOARD);
    camWidth = 640;
    camHeight = 480;
    evmEnable = false;
    //Size of the Image used for recognition (Square)
    finalSize = 150;
    
    cam.initGrabber(camWidth, camHeight);
    
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
    loadFaceImages();
    person = -1;
    
    //create first tracker
    camTracker.setup();
    
    //result
    faceResult.setup(camWidth, camHeight, finalSize);
    
    //load recognition data
    recognizer.learn();
    
#ifdef USE_UVC_CONTROLS
    uvcControl.useCamera(0x5ac, 0x8507, 0x00);
#endif
    
}

//--------------------------------------------------------------
void faceRecognition::update(){
    //update EVM parameters
    if(evmEnable){
        evm.setTemporalFilter(filter);
        evm.setParamsIIR(alpha_iir, lambda_c_iir, r1, r2, chromAttenuation_iir);
        evm.setParamsIdeal(alpha_ideal, lambda_c_ideal, fl, fh, samplingRate, chromAttenuation_ideal);
    }
    
    cam.update();
    if(cam.isFrameNew()) {
        
        /*
         * PART 1 - Face Detection and equalization
         * 1) Detect face with ofxFaceTracker
         * 2) Get the Face Img
         * 3) Convert to Grayscale and Histogram equalize
         */
        bool faceFound = findFace(cam.getPixelsRef(), faceResult);
        
        /*
         * PART 2 Recognition
         *
         */
        
        if(recognizer.isTrained() && doRecognize){
            person = -1;
            person = recognizer.recognize(faceResult.faceCvGray);
            faceResult.person = person;
            if(person != -1){
                double lsd = recognizer.getLeastDistSq();
                if(lsd < faceResult.recLeastSquareDist){
                    faceResult.recLeastSquareDist = lsd;
                    faceResult.bestPerson = person;
                }
            }
        }
        
        /*
         * Pulse detection
         * TODO: get it working
         */
        
        if(evmEnable)
            evm.update(toCv(faceResult.faceImage));
    }
}


//----- Finding the face with face tracker, copying etc. -----
bool faceRecognition::findFace(ofImage img, ofxFaceTrackerResult &result){
    
    
    camTracker.update(toCv(img));    
    bool _faceFound = camTracker.getFound();
    
    if(_faceFound)
    {
        result.faceFound = true;
        if(!result.faceNewFound)
        {
            result.faceId++;
            result.faceNewFound = true;
        }
        
        if(!result.initialized)
            result.setup(camWidth, camHeight, finalSize);
        
        //------------------
        // Here we start by masking the face out of the camera image
        //------------------
        result.faceOutline = camTracker.getImageFeature(ofxFaceTracker::FACE_OUTLINE);
        result.faceBB = result.faceOutline.getBoundingBox();
        result.faceSolid.clear();
        
        for(int i=0; i < result.faceOutline.getVertices().size(); i++) {
            if(i == 0) {
                result.faceSolid.newSubPath();
                result.faceSolid.moveTo(result.faceOutline.getVertices()[i]);
            }else{
                result.faceSolid.lineTo(result.faceOutline.getVertices()[i]);
            }
        }
        
        result.faceSolid.close();
        
        result.maskFbo.begin();
        ofClear(0, 255);
        result.faceSolid.draw();
        result.maskFbo.end();
        
        result.resultFbo.begin();
        ofClear(0, 0, 0, 0);
        maskShader.begin();
        maskShader.setUniformTexture("maskTex", result.maskFbo.getTextureReference(), 1);
        img.draw(0, 0);
        maskShader.end();
        result.resultFbo.end();
        
        //------------------
        // Resulting Image from FBO cropped and resized
        //------------------
        ofPixels pix;
        result.resultFbo.readToPixels(pix);
        result.faceImage.setFromPixels(pix);
        result.faceImage.crop(result.faceBB.position.x, result.faceBB.position.y, result.faceBB.width, result.faceBB.height);
        result.faceImage.resize(finalSize, finalSize);
        
        result.faceCvColor.setFromPixels(result.faceImage.getPixels(), result.faceImage.getWidth(), result.faceImage.getHeight());
        result.faceCvGray = result.faceCvColor;
        cvEqualizeHist(result.faceCvGray.getCvImage(), result.faceCvGray.getCvImage());
        
        if(!result.eyeFound)
            findEyeColor(img);
        if(!result.skinFound)
            findSkinColor(img);
        
    }else{
        result.faceFound = false;
        result.faceNewFound = false;
        result.eyeFound = false;
        faceResult.skinFound = false;
        result.recLeastSquareDist = std::numeric_limits<double>::max();
        result.bestPerson = -1;
    }
    
    return _faceFound;
}

//--------------------------------------------------------------
void faceRecognition::findEyeColor(ofImage img){
    
    // Eye Color
    //cv::Mat eyeGray;
    cv::Mat eyeCanny;
    cv::Mat eyeThresh;
    vector<vector<cv::Point> > contours;
    vector<Vec4i> hierarchy;
    Point2f com;
    
    ofPolyline leftEyeOutline = camTracker.getImageFeature(ofxFaceTracker::LEFT_EYE);
    ofRectangle leftEyeBB = leftEyeOutline.getBoundingBox();
    faceResult.eyeImage.cropFrom(img, leftEyeBB.x - 10.f, leftEyeBB.y - 10.f, leftEyeBB.width + 20.f, leftEyeBB.height + 20.f);
    
    convertColor(faceResult.eyeImage.getPixelsRef(), eyeCanny, CV_RGB2HSV);
    cv::inRange(eyeCanny, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 80), eyeThresh);
    blur(eyeThresh, eyeThresh, cv::Size(3,3));
  
    /// Find contours
    findContours( eyeThresh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    
    /// Get the moments
    vector<Moments> mu(contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    { mu[i] = moments( contours[i], false ); }
    
    ///  Get the mass centers:
    vector<Point2f> mc( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }
    
    //Total CoM
    for( int i = 0; i < mc.size(); i++ )
    { com += mc[i]; }
    com.x = com.x / mc.size();
    com.y = com.y / mc.size();
    
    if(mc.size() > 0){
        faceResult.eyeFound = true;
        
        com.x = com.x / mc.size();
        com.y = com.y / mc.size();
        
        com = mc[0];
        faceResult.eyePosition.x = com.x + leftEyeBB.x - 10.f;
        faceResult.eyePosition.y = com.y + leftEyeBB.y - 10.f;
        //when we find circle push it to the results and calculate the color
        //sample the color around the found center -
        //TODO: this needs to be done another way
        float offset = 2.0f;
        
        int avrColor = averageColorOfArea(faceResult.eyeImage.getPixelsRef(), com.x, com.y, 5, 5, faceResult.eyeImage.width, faceResult.eyeImage.height);
        
        faceResult.eyeColor.setHex(avrColor);
        
    }
}

//--------------------------------------------------------------
void faceRecognition::findSkinColor(ofImage img){
    //2 und 30
    ofVec2f l, c;
    l = camTracker.getImagePoint(2);
    c = camTracker.getImagePoint(30);
    float dist = l.distance(c);
    
    int avrColor = averageColorOfArea(img.getPixelsRef(), c.x - dist / 2, c.y, 10, 10, img.width, img.height);
    faceResult.skinColor.setHex(avrColor);
    faceResult.skinFound = true;
}


int faceRecognition::averageColorOfArea(ofPixels pixels, int cx, int cy, int avgW, int avgH, int imgW, int imgH){
	//COPIED from forum
    //average a certain area around a point in a pixel area
	//return the resulting averaged color in hex/int	
	int rr = 0;
	int gg = 0;
	int bb = 0;
	float cnt=0;
	
	for (int xx = cx-avgW; xx <= cx+avgW; xx++){
		for (int yy = cy-avgH; yy <= cy+avgH; yy++){
			
			int cxx = xx;
			int cyy = yy;
			if(xx < 0) cxx = 0;
			if(xx >= imgW) cxx = imgW-1;
			if(yy < 0) cyy = 0;
			if(yy >= imgH) cyy = imgH-1;
			
			rr = rr + pixels[(cyy * imgW*3) + cxx * 3 + 0];
			gg = gg + pixels[(cyy * imgW*3) + cxx * 3 + 1];
			bb = bb + pixels[(cyy * imgW*3) + cxx * 3 + 2];
			cnt++;
			
		}
	}
	
	rr = int(rr/cnt);
	gg = int(gg/cnt);
	bb = int(bb/cnt);
	
	int lookUpIdx = (rr << 16) + (gg << 8) + bb;
	
	return lookUpIdx;
}



//--------------------------------------------------------------
void faceRecognition::draw(){
    
    ofBackgroundGradient(ofColor(64), ofColor(0));
    ofSetColor(255, 255, 255);
    
    ofDrawBitmapStringHighlight("CAM STREAM " + ofToString(ofGetFrameRate()), 20.f, 40.f);
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
    
    if(evmEnable){
      ofSetColor(255, 255, 255);
        ofDrawBitmapStringHighlight("MOTION AMPLIFIED", camWidth + 50, 440);
        ofSetColor(255, 0, 0);
        ofRect(camWidth + 50, 450, 170, 170);
    }
    
    
    if(faceResult.faceFound) {
        ofSetColor(255, 255, 255);
        faceResult.faceCvColor.draw(camWidth + 60, 60);
        faceResult.faceCvGray.draw(camWidth + 60, 260);
        
        faceResult.eyeImage.draw(camWidth + 60, 440);        
        //drawMat(eyeCanny, camWidth + 60, 500);
        //drawMat(eyeThresh, camWidth + 60, 560);
        
        if(faceResult.eyeFound){
            ofSetColor(faceResult.eyeColor);
            ofRect(camWidth + 200, 440.f, 50.f, 20.f);
            ofDrawBitmapString(ofToString( faceResult.eyeColor.getSaturation() ), camWidth + 250, 420.f);
            //ofCircle(faceResult.eyePosition.x + 20.f, faceResult.eyePosition.y + 60.f, 8);
            ofSetColor(255, 255, 255);
        }
        
        if(evmEnable)
            evm.draw(camWidth + 60, 460);
        
        ofPushMatrix();
        ofTranslate(20.f, 60.f);
        camTracker.draw();
        ofPopMatrix();
        
        //skin color
        ofSetColor(faceResult.skinColor);
        ofRect(camWidth + 200, 550.f, 50.f, 20.f);
        ofDrawBitmapString(ofToString( faceResult.skinColor.getLightness() ), camWidth + 250, 560.f);
        
        ofDrawBitmapStringHighlight(" ID: " + ofToString(faceResult.faceId), faceResult.faceBB.x, faceResult.faceBB.y);
    }
    
    if(person != -1 && faceResult.faceFound){
        ofDrawBitmapStringHighlight("Person found: " + ofToString(faceResult.person) + " Best Match " + ofToString(faceResult.bestPerson), 400.f, camHeight + 85.f);
        ofDrawBitmapStringHighlight("Name: " + getPersonName(faceResult.bestPerson), faceResult.faceBB.x, faceResult.faceBB.y + 20.f);
        recognizer.drawPerson(faceResult.bestPerson, 400, camHeight + 95);
    }
    
}
//--------------------------------------------------------------
void faceRecognition::loadFaceImages(){
    string line;
    ifstream fin(ofToDataPath(filename).c_str());
    if (fin.is_open())
    {
        while ( fin.good() )
        {
            getline (fin,line);
            //cout << line << endl;
            trainingImages.push_back(line);
        }
        fin.close();
    }
}

//--------------------------------------------------------------
void faceRecognition::saveFaceImage(ofxCvGrayscaleImage img){
    ofImage saveImg;
    saveImg.setFromPixels(img.getPixels(), img.width, img.height, OF_IMAGE_GRAYSCALE);
    
    int numFound = 0;
    string imgName = uiFirstName->getTextString() + "_" + uiLastName->getTextString() + "_" + ofToString(numFound) + ".jpg";
    
    for(int x=0; x<trainingImages.size(); x++){
        if(trainingImages[x] == imgName){
            numFound++;
            imgName = uiFirstName->getTextString() + "_" + uiLastName->getTextString() + "_" + ofToString(numFound) + ".jpg";
        }            
    }
    
    trainingImages.push_back(imgName);
    //FirstName_LastName . fileFormat
    //Evtl noch XML datei / bzw txt datei mit mehr daten
    ofstream fout(ofToDataPath(filename).c_str(), fstream::in | fstream::out | fstream::app);
    fout << imgName << endl;
    fout.close();
    
    string saveFileName = "faces/";
    saveFileName += imgName;
    
    cout << "saving to: " << saveFileName << endl;
    
    saveImg.saveImage(saveFileName);
}

//--------------------------------------------------------------
string faceRecognition::getPersonName(int _id){
    string name = trainingImages[_id];
    string firstName = "";
    string lastName = "";
    //TODO: split up string
    string::size_type position = name.find('_');
    firstName = name.substr(0, position);
    
    string::size_type position2 = name.rfind('_');
    lastName = name.substr(position + 1, name.length() - position2);
    
    name = firstName + " " + lastName;
    
    return name;
}

//--------------------------------------------------------------
void faceRecognition::guiEvent(ofxUIEventArgs &e){
    string name = e.widget->getName();
	int kind = e.widget->getKind();
    
    if(name == "Temporal IIR"){
        filter = EVM_TEMPORAL_IIR;
    }
    
    if(name == "Temporal Ideal (Unimplemented)"){
        filter = EVM_TEMPORAL_IDEAL;
    }
    
    if(name == "Enable EVM"){
        uiEvm->toggleVisible();
    }
    
    if(name == "save"){
        ofxUIButton *btn = (ofxUIButton *) e.widget;
        bool pressed = btn->getValue();
        if(uiFirstName->getTextString().length() == 0 &&
           uiLastName->getTextString().length() == 0 &&
           pressed)
        {
            uiFeedback->setLabel("Enter name first!");
        }else if(!pressed){
            saveFaceImage(faceResult.faceCvGray);
            uiFeedback->setLabel("Image saved!");
        }
    }
    
    if(name == "learn"){
        ofxUIButton *btn = (ofxUIButton *) e.widget;
        bool pressed = btn->getValue();
        
        if(!pressed && !recognizer.isTrained()){
            uiFeedback->setLabel("Learning training data!");
            recognizer.learn();
        }else{
            uiFeedback->setLabel("Allready trained!");
        }
    }
    
    cout << "UI event: " << name << " kind: " << kind << endl;
}

//--------------------------------------------------------------
void faceRecognition::keyPressed(int key){

}

//--------------------------------------------------------------
void faceRecognition::keyReleased(int key){
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
    uiEnableEvm = new ofxUIToggle("Enable EVM", &evmEnable, 20.f, 20.f);
    uiDoRecognize = new ofxUIToggle("Do Recognize", &doRecognize, 20.f, 20.f);
    
    uiCanvas->addLabel("DATA INPUT", OFX_UI_FONT_MEDIUM);
    uiCanvas->addWidgetDown(uiFeedback);
    uiCanvas->addSpacer();
    uiCanvas->addWidgetDown(uiFirstName);
    uiCanvas->addWidgetDown(uiLastName);
    uiCanvas->addSpacer();
    uiCanvas->addButton("save", false);
    uiCanvas->addSpacer();
    uiCanvas->addWidgetDown(uiEnableEvm);
    uiCanvas->addSpacer();
    uiCanvas->addButton("learn", false);
    uiCanvas->addWidgetRight(uiDoRecognize);
    
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
    
    uiEvm->toggleVisible();
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