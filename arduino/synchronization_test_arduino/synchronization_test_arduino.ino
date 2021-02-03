/* Use a photoresistor (or photocell) to turn on an LED in the dark
   More info and circuit schematic: http://www.ardumotive.com/how-to-use-a-photoresistor-en.html
   Dev: Michalis Vasilakis // Date: 8/6/2015 // www.ardumotive.com */
   

//Constants
const int pResistor = A0; // Photoresistor at Arduino analog pin A0
//const int ledPin=9;       // Led pin at Arduino pin 9
const int pwm = 2 ;     //initializing pin 2 as ‘pwm’ variable
const int LEDPin = 13;
const int pulse_debouncer_steps = 20;
//Variables
int value;          // Store value from photoresistor (0-1023)
int value_last;
int pulse_debouncer = pulse_debouncer_steps;


void setup(){
 Serial.begin(9600);
 pinMode(pResistor, INPUT);// Set pResistor - A0 pin as an input (optional)
 pinMode(pwm,OUTPUT);
 pinMode(LEDPin,OUTPUT);

 value_last = 0;
}

void loop(){
  
  if (pulse_debouncer < pulse_debouncer_steps) {
    analogWrite(pwm,25);
    pulse_debouncer += 1;
  }
  else {
    analogWrite(pwm,0) ;
    digitalWrite(LEDPin, LOW);
  }
  
  value = analogRead(pResistor);
  //Serial.println(value);
  
  if (value - value_last > 200 && pulse_debouncer >= pulse_debouncer_steps) {
    Serial.println("Pulse detected");
    digitalWrite(LEDPin, HIGH);
    analogWrite(pwm,25);
    pulse_debouncer = 0;
  }
  //Serial.print("\n");
  //You can change value "25"


  
//  if (value > 25){
//    digitalWrite(ledPin, LOW);  //Turn led off
//    // Serial.print("turn OFF");
//  }
//  else{
//    digitalWrite(ledPin, HIGH); //Turn led on
//    //Serial.print("turn ON");
//  }

  delay(10); //Small delay
}
