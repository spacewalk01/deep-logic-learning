def nt(*X):
    return sum(X)

import itertools


def _provide_scholaship1(high_gpa, sent_application, take_course, has_supervisor):
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

    provide_scholarship = _provide_scholaship1(high_gpa, sent_application, take_course, has_supervisor)

    return provide_scholarship


def predicate(grad_student, high_gpa, sent_application, freshman, complete_course):
    has_supervisor = 0
    if 13.437 < 4.60384 * nt(grad_student, sent_application, high_gpa) + -5.40128 * nt(complete_course, freshman):
        has_supervisor = 1
    take_course = 0
    if 13.4306 < 4.60164 * nt(grad_student, sent_application, high_gpa) + -5.39673 * nt(complete_course, freshman):
        take_course = 1
    head1 = 0
    if -7.35178 < -3.96283 * nt(grad_student, sent_application, high_gpa) + 4.50825 * nt(complete_course, freshman):
        head1 = 1
    head2 = 0
    if -7.36688 < 4.51598 * nt(complete_course, freshman) + -3.97372 * nt(grad_student, sent_application, high_gpa):
        head2 = 1
    head3 = 0
    if -7.23316 < 4.47956 * nt(complete_course, freshman) + -3.91775 * nt(grad_student, sent_application, high_gpa):
        head3 = 1
    scholarship_candidate = 0
    if 2.25758 < 11.5396 * nt(has_supervisor, take_course) + -7.36303 * nt(head1, head2, head3):
        scholarship_candidate = 1

    print('grad_s', grad_student, 'high_gpa', high_gpa, 'sent_app', sent_application, 'freshman', freshman, '수료',complete_course,
          'take_course=', take_course, 'has_supervisor=', has_supervisor, 'provide_scholarship=',scholarship_candidate)
    return scholarship_candidate
combo = list(itertools.product([0, 1], repeat=5))
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
y_ = []

i = 0
correct = 0.0

for c in combo:

    grad_student = c[0]
    high_gpa = c[1]
    sent_application = c[2]
    freshman = c[3]
    complete_course = c[4]

    pred = predicate(grad_student, high_gpa, sent_application, freshman, complete_course)
    y = estimate(grad_student, high_gpa, sent_application, freshman, complete_course)

    i += 1
    if pred == y:
        correct += 1.0
    print(y, pred)
print('Training Accuracy: ' + str(100 * correct / i) + '%')

