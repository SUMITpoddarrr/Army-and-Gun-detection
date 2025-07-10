#Threat Detection System Using YOLOv8
This project is an AI-powered surveillance system that automatically detects firearms in images and determines whether the weapon is carried by a civilian or a uniformed officer (army/police). If a civilian is found carrying a gun, an alert is triggered and the annotated image is saved separately.

Key Features
-Gun Detection using YOLOv8-Nano model
-Person Classification:

Class 0 → Army/Police (uniformed)

Class 1 → Civilians (normal dress)



#Models Used
1. Gun Detection Model
File: best_m_gun.pt

Framework: YOLOv8-Nano

Output: Bounding boxes of detected guns

Class:

0 → Gun

2. Person Classification Model
File: best.pt

Framework: YOLOv8

Output: Person classification

Classes:

0 → Army/Police (uniformed)

1 → Civilian (normal dress)

#Threat Detection Logic
For each image:

Detect guns using the gun model.

If no guns are found → skip the image.

If guns are found:

Detect people using the uniform model.

Calculate IoU (Intersection over Union) between each gun box and civilian box.

If any civilian overlaps with a gun (> 0.01 IoU), mark the image as an alert.

Annotate and save accordingly:

Civilians with guns → alert/

Only officers with guns → normal/

# How to Run
1. Clone the repository

git clone https://github.com/SUMITpoddarrr/Army-and-Gun-detection.git
cd threat-detection-system
2. Install dependencies

pip install -r requirements.txt
3. Prepare the folders and add models
Place your input images inside the images/ folder.

Place the following models:

best_m_gun.pt → in the project root

best.pt (for uniform/civilian detection) → update the path in code if needed

4. Run the detection
python main.py

 #Output Example
-Green → Uniformed person

-Yellow → Civilian carrying a gun (alert)

-Red → Gun detected

Annotated images are automatically saved in:

normal/ → No threat

alert/ → Civilian detected with gun
