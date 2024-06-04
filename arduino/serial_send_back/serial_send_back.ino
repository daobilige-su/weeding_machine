int inByte = 0;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.print("Hello\n");
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available())
  {
    // get incoming byte:
    inByte = Serial.read();         

    // send the same character back to serial port
    Serial.write(inByte);
  }
}
