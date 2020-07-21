import cv2
import numpy as np
import os, time, sys, math

class Vehicle(object):
    def __init__(self, id, position):
        self.id = id
        self.positions = [position]
        self.frames_since_seen = 0
        self.counted = False
        
    def last_position(self):
        return self.positions[-1]
        
    def add_position(self, new_position):
        self.positions.append(new_position)
        self.frames_since_seen = 0

class VehicleCounter(object):
    def __init__(self, shape, divider):
        self.height, self.width = shape
        self.divider = divider

        self.vehicles = []
        self.next_vehicle_id = 0
        self.vehicle_count = 0
        self.max_unseen_frames = 40 #threshold
    
    @staticmethod
    def get_vector(a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.

        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values increase in clockwise direction.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])

        distance = math.sqrt(dx**2 + dy**2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx/dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx/dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx/dy))
            else:
                angle = 180.0        

        return distance, angle 

    @staticmethod
    def is_valid_vector(a):
        distance, angle = a
        threshold_distance = max(50.0, -0.008 * angle**2 + 0.4 * angle + 60.0)
        return (distance <= threshold_distance)

    def update_vehicle(self, vehicle, matches):
        for i, match in enumerate(matches):
            contour, centroid = match
            
            vector = self.get_vector(vehicle.last_position(), centroid)
            if self.is_valid_vector(vector):
                vehicle.add_position(centroid)
                return i
        
        vehicle.frames_since_seen += 1
        return None
        
    def update_count(self, matches, processed):
        #ToDo: Add checking if coords is almost the same [completed]
        
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches)
            if i is not None:
                del matches[i]
        
        for match in matches:
            contour, centroid = match
            new_vehicle = Vehicle(self.next_vehicle_id, centroid)
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)
        # print(self.vehicles)
        
        for vehicle in self.vehicles:
            if not vehicle.counted and (vehicle.last_position()[0] > self.divider) :
                self.vehicle_count += 1
                vehicle.counted = True
        
        for vehicle in self.vehicles:
            #print(vehicle.id)
            #print(vehicle.positions)
            cv2.circle(processed, (vehicle.positions[0]), 2, (255,0,255), 2, 8, 0)
            #cv2.putText(processed, str(vehicle.id) , vehicle.positions[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        removed = [ v.id for v in self.vehicles if v.frames_since_seen >= self.max_unseen_frames ]
        self.vehicles[:] = [ v for v in self.vehicles if not v.frames_since_seen >= self.max_unseen_frames ]
        #for id in removed:
        #    print(id)
# ============================================================================



IMAGE_DIR = 'images'
frame_number = -1

VehCount = 0
debugint = 240

def findCenter(x,y,w,h):
    cx = int((x+w)/2)
    cy = int((y+h)/2)
    return cx,cy

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return (cx, cy)

# =========================================================================
def save_frame(file_name_format, frame_number, frame, label_format):
    file_name = file_name_format % frame_number
    label = label_format % frame_number
    cv2.imwrite(file_name, frame)
    
def filter_mask(fg_mask):
    #fg_mask = cv2.GaussianBlur(fg_mask,(1,1),0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Fill 
    closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate 
    dilation = cv2.dilate(opening, kernel, iterations = 3)

    return dilation
    
def detect_vehicles(fg_mask):

    MIN_CONTOUR_WIDTH = 65
    MIN_CONTOUR_HEIGHT = 65

    # Vehicle contour in image
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    matches = []
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT)

        if not contour_valid:
            continue

        centroid = get_centroid(x, y, w, h)
        #centroid = findCenter(x, y, w, h)

        matches.append(((x, y, w, h), centroid))
    
    return matches

    
def process_frame(frame_number, frame, bg_subtractor, car_counter):
    global VehCount
    global debugint
    # Create a copy of source frame to draw into
    processed = frame.copy()

    # Line untuk Ngitung
    #cv2.line(processed, (0, int(car_counter.divider) + debugint), (frame.shape[1], int(car_counter.divider) + debugint), (255, 255, 0), 1)
    #cv2.line(processed , (0, 180), (frame.shape[1], 180), (255,255,0),1)
    
    #print("bates",(0, int(car_counter.divider) + 60), (frame.shape[1], int(car_counter.divider) + 60))
    
    # Remove the background
    fg_mask = bg_subtractor.apply(frame, None, 0.007)
    fg_mask = filter_mask(fg_mask)
    #fg_mask = filter_mask(fg_mask)

    matches = detect_vehicles(fg_mask)
    
    for (i, match) in enumerate(matches):
        contour, centroid = match
        x, y, w, h = contour
        
        cv2.rectangle(processed, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 1)
        cv2.circle(processed, centroid, 2, (0, 0, 255), -1)
    
    car_counter.update_count(matches, processed)
    
    
    #cv2.putText(processed, 'Count: ' + str(len(car_counter.vehicles)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0), 2, cv2.LINE_AA)
    return processed
# =========================================================================


bgsMOG = cv2.bgsegm.createBackgroundSubtractorMOG()
capture = cv2.VideoCapture('Sample3.mp4')

if capture:
    #default_bg = cv2.imread('images/original/original_%04d.png' % 0)
    #cv2.imshow('BACKGROUND',default_bg)
    
    car_counter = None
    
    while True:
        frame_number += 1
        ret, frame = capture.read()
        if ret:
            fgmask = bgsMOG.apply(frame, None, 0.01)
            cv2.imshow('MaskBG',fgmask)
            
            fgmask = filter_mask(fgmask)
            cv2.imshow('MaskBG V2',fgmask)
            
            if car_counter is None:
                car_counter = VehicleCounter(frame.shape[:2], frame.shape[0] / 2)
            
            processed = process_frame(frame_number, frame, bgsMOG, car_counter)

            cv2.putText(processed, 'Count: ' + str(car_counter.vehicle_count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Original',frame)
            cv2.imshow('Processed',processed)
            save_frame(IMAGE_DIR + "/processed_%04d.png", frame_number, processed, "processed frame #%d")
            save_frame(IMAGE_DIR + "/mask/mask_%04d.png", frame_number, fgmask, "foreground mask for frame #%d")
            save_frame(IMAGE_DIR + "/original/original_%04d.png", frame_number, frame, "processed frame #%d")
            
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
capture.release()
cv2.destroyAllWindows()