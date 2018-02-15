import smtplib
 
def sendemail(from_addr, to_addr_list, cc_addr_list,
              subject, message,
              login, password,
              smtpserver='smtp.gmail.com:587'):
    header  = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list)
    header += 'Subject: %s\n\n' % subject
    message = header + message
 
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()

def main():
  sendemail(from_addr    = 'dronemasterprosjekt@gmail.com', 
          to_addr_list = ['krishk@stud.ntnu.no'],
          cc_addr_list = ['adriari@stud.ntnu.no'], 
          subject      = 'Testing python mail', 
          message      = 'Howdy from a python function', 
          login        = 'dronemasterprosjekt', 
          password     = 'drone123')

if __name__ == '__main__':
  main()  