import sys

def error_message_datails(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    # Get the file name where error has occured
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Get the error messsage
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_datails(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    

# if __name__ == "__main__":

#     try:
#         a = 0/0
#     except Exception as e:
#         raise CustomException(e,sys)