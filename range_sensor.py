from gpiozero import DistanceSensor   # Uses built-in software trigger
sensor = DistanceSensor(echo=24, trigger=23, max_distance=2.0)  # metres

def distance_cm():
    return round(sensor.distance * 100, 1)      # cm
