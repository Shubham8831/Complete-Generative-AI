from pydantic import BaseModel
from typing import Optional


class Student(BaseModel):

    name : str
    name : str = "Shubham" # passing shubham as a default value
    age : Optional[int] = None # default value is none
    # email : EmailStr 


new_student = {
    "name": "nitish",
    "age": "35" # pydantic can also do implicit type conversion str -> int 
}

student = Student(**new_student)

print(student)
print(type(student))