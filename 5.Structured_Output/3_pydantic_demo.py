from pydantic import BaseModel, Field
from typing import Optional

#Field function: default value, constraints, description, regex expression

class Student(BaseModel):

    name : str
    name : str = "Shubham" # passing shubham as a default value
    age : Optional[int] = None # default value is none
    # email : EmailStr # this is buildin validation
    cgpa : float = Field(gt=0, lt=10, default=5, description="Adecimal vaule representing the cgps of the student.")


new_student = {
    "name": "nitish",
    "age": "35", # pydantic can also do implicit type conversion str -> int
    "cgpa": 11 # this will give error as cgps value is set in range 0-10 by field fn.
}

student = Student(**new_student)

print(student) # this will be in pydantic
student_dict = dict(student) # converting it to dictionary
print(student_dict["age"])

student_json = student.model_dump_json() # converting it to json