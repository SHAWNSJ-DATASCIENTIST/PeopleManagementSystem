import machine
import utime

# Hardware stabilization
utime.sleep(1)

piezo = machine.ADC(28)
onboard_led = machine.Pin("LED", machine.Pin.OUT)
led1 = machine.Pin(13, machine.Pin.OUT)
led2 = machine.Pin(14, machine.Pin.OUT)
led3 = machine.Pin(15, machine.Pin.OUT)
red_led = machine.Pin(0, machine.Pin.OUT)

# STARTUP SIGNAL
for _ in range(3):
    onboard_led.value(1)
    led1.value(1)
    led2.value(1)
    led3.value(1)
    red_led.value(1)
    utime.sleep(0.1)
    onboard_led.value(0)
    led1.value(0)
    led2.value(0)
    led3.value(0)
    red_led.value(0)
    utime.sleep(0.1)

threshold = 5000 

while True:
    raw_val = piezo.read_u16()
    
    # Calculate voltage every iteration
    # $V = \frac{raw\_val \times 3.3}{65535}$
    voltage = (raw_val * 3.3) / 65535
    
    # Print formatted voltage
    print(f"Voltage: {voltage:.2f} V")
    
    if raw_val > threshold:
        onboard_led.value(1)
        if voltage > 0.5: led1.value(1)
        if voltage > 1.5: led2.value(1)
        if voltage > 2.5: led3.value(1)
        if voltage > 2.7: red_led.value(1)
            
        utime.sleep(0.1)
        
        onboard_led.value(0)
        led1.value(0)
        led2.value(0)
        led3.value(0)
        red_led.value(0)
        utime.sleep(0.1)
    
    utime.sleep(0.01)