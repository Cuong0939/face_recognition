# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import stuff dependencies
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# IMport other dependenies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np



#Build app and layout
class CamApp(App):
    def build(self):
        #Main layput components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text='verify',on_press=self.verify,size_hint=(1,.1))
        self.verification_label = Label(text='Verification Uninitiated',size_hint=(1,.1))
        
        #Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)


        # Load keras model
        self.model = tf.keras.models.load_model('app/siamesemodelv2.h5',custom_objects = {'L1Dist':L1Dist})
        
        # Setup video capture device
        self.capture = cv2.VideoCapture(0)

        Clock.schedule_interval(self.update,1.0/33.0)


        return layout
    
    #Run continuosly to get webcam feed
    def update(self,*agrs):
        
        #Read frame from opencv
        ret,frame = self.capture.read()
        frame = frame[150:150+250,200:200+250,:]

        # Flip horizontall and covert image to texture
        buf = cv2.flip(frame,0).tostring()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt = 'bgr')
        img_texture.blit_buffer(buf,colorfmt ='bgr',bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def preprocess(self,file_path):
        #Read in image from file path
        byte_img = tf.io.read_file(file_path)
        #load in the image
        img =tf.io.decode_jpeg(byte_img)

        # Preprocessing steps - resizing the image to be  100x100x3
        img = tf.image.resize(img,(100,100))
        # Scale image to be between 0 and 1
        img = img / 255.0

        return img

    #Verification function to verify person
    def verify(self,*args):
        # Specify threshold
        detection_threshold = 0.99
        verification_threshold = 0.8

        # Captur input image from webcam
        SAVE_PATH =os.path.join('app/application_data/','input_image/','input_image.jpg')
        ret,frame =   self.capture.read()       
        frame = frame[150:150+250,200:200+250,:]
        cv2.imwrite(SAVE_PATH,frame)

        results = []
        for image in os.listdir(os.path.join('app/application_data/','verification_images/')):
            input_img = self.preprocess(os.path.join('app/application_data/','input_image/','input_image.jpg'))
            validation_img = self.preprocess(os.path.join('app/application_data/','verification_images/',image))

            #make preditions
            result = self.model.predict(list(np.expand_dims([input_img,validation_img],axis = 1)))
            results.append(result)
        # Detection Threshold: Metric above which a prediciton is considered positive
        detection = np.sum(np.array(results) > detection_threshold)

        #Verification threshold: Proportion of positive predictions/total positive samples
        verification = detection / len(os.listdir(os.path.join('app/application_data/','verification_images/')))
        verified = verification > verification_threshold

        # Set verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # Log out details
        Logger.info(results)
        Logger.info(verified)


        return results, verified
if __name__ == '__main__':
    CamApp().run()