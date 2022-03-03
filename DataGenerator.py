import random as rand
import operator

def _scholarship_candidate(high_gpa, sent_application, take_course, has_supervisor):
    if high_gpa == 1 and sent_application == 1 and take_course == 1 and has_supervisor == 1:
        return 1
    else:
        return 0

def _has_supervisor(grad_student, freshman):
    if grad_student == 1 and freshman == 0:
        return 1
    else:
        return 0

def _take_course(grad_student, complete_course):
    if grad_student == 1 and complete_course == 0:
        return 1
    else:
        return 0

def estimate(grad_student, high_gpa, sent_application, freshman, complete_course):

    take_course = _take_course(grad_student, complete_course)
    has_supervisor = _has_supervisor(grad_student, freshman)

    provide_scholarship = _scholarship_candidate(high_gpa, sent_application, take_course, has_supervisor)

    return provide_scholarship

import itertools
combo = list(itertools.product([0, 1], repeat=5))

result = []
for c in combo:

    grad_student = c[0]
    high_gpa = c[1]
    sent_application = c[2]
    freshman = c[3]
    complete_course = c[4]

    y = estimate(grad_student, high_gpa, sent_application, freshman, complete_course)

    if y == 1:
        result.append(1)
    else:
        result.append(0)


    print(str(grad_student) + ',' + str(high_gpa) + ',' + str(sent_application) + ',' + str(freshman)
          + ',' + str(complete_course) + ',' + str(y))




print(result)