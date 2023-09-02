from multiprocessing import Pool

# def f(x):
#     return x*x
#
#
# # if __name__ == '__main__':
# #     with Pool(5) as p:
# #         print(p.map(f, [1, 2, 3]))

# from multiprocessing import Process
#
# def f(name):
#     print('John how are you', name)
#
# if __name__ == '__main__':
#     print("JOHN")
#     p = Process(target=f, args=('bob',))
#     p.start()
#     p.join()


# from multiprocessing import Process
# import os
#
# def info(title):
#     print(title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())
#
# def f(name):
#     info('function f')
#     print('hello', name)
#
# if __name__ == '__main__':
#     info('main line')
#     p = Process(target=f, args=('bob',))
#     p.start()
#     p.join()


# from multiprocessing import Process
# class print_data:
#     def __init__(self):
#         pass
#
#     def print_func(self,continent='Asia'):
#         print('The name of continent is : ', continent)
#
# def print_func(continent='Asia'):
#     print('The name of continent is : ', continent)
#
# if __name__ == "__main__":  # confirms that the code is under main function
#     print("John")
#     names = ['America', 'Europe', 'Africa']
#     procs = []
#     test_object = print_data()
#     proc = Process(target=test_object.print_func)  # instantiating without any argument
#     procs.append(proc)
#     proc.start()
#
#     # instantiating process with arguments
#     for name in names:
#         # print(name)
#         proc = Process(target=print_func, args=(name,))
#         procs.append(proc)
#         proc.start()
#         # print(name)
#
#     # complete the processes
#     for proc in procs:
#         proc.join()

# from multiprocessing import Lock, Process, Queue, current_process
# import time
# import queue # imported for using queue.Empty exception
#
#
# def do_job(tasks_to_accomplish, tasks_that_are_done):
#     while True:
#         try:
#             '''
#                 try to get task from the queue. get_nowait() function will
#                 raise queue.Empty exception if the queue is empty.
#                 queue(False) function would do the same task also.
#             '''
#             task = tasks_to_accomplish.get_nowait()
#         except queue.Empty:
#
#             break
#         else:
#             '''
#                 if no exception has been raised, add the task completion
#                 message to task_that_are_done queue
#             '''
#             print(task)
#             tasks_that_are_done.put(task + ' is done by ' + current_process().name)
#             time.sleep(.5)
#     return True
#
#
# def main():
#     number_of_task = 10
#     number_of_processes = 4
#     tasks_to_accomplish = Queue()
#     tasks_that_are_done = Queue()
#     processes = []
#
#     for i in range(number_of_task):
#         tasks_to_accomplish.put("Task no " + str(i))
#
#     # creating processes
#     for w in range(number_of_processes):
#         p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done))
#         processes.append(p)
#         p.start()



# import multiprocessing as mp
#
# def foo(q):
#     q.put('hello')
#
# if __name__ == '__main__':
#     ctx = mp.get_context('spawn')
#     q = ctx.Queue()
#     p = ctx.Process(target=foo, args=(q,))
#     p.start()
#     print(q.get())
#     p.join()


from multiprocessing import Process, Queue

def f(q):
    q.put([42, None, 'hello'])

if __name__ == '__main__':
    q = Queue()
    f(q)

    p = Process()
    p.start()
    print(p.pid)
    print("John")
    # print(q.get())
    # prints "[42, None, 'hello']"
    p.join()