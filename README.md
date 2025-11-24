
 # pip install flet opencv-python Pillow ultralytics numpy
Clone the repository
# git clone https://github.com/kiew56/hackathon.git
# cd HealAI
Assets

Running on Desktop
# python main.py
Launches the Flet app in your default browser.
Use Scan a Video to detect injuries in real-time using the camera.
Installing on Android (APK)
Install Flet’s Android export tools:
# pip install flet
Export your app to APK:
# flet build apk 
This will generate an APK in the android_build folder.
Transfer the APK to your Android device and install it.
Optional: Enable installation from unknown sources on your Android device.
The APK includes all dependencies, your YOLO model, and assets, so the app can run offline on the mobile device.
