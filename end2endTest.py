import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import threading
import time
import gui  # Assuming gui has the GUI class and a shutdown function

# helper function to help deal with nicegui
def uncheck_all_checkboxes(driver):
    checkboxes = driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
    for checkbox in checkboxes:
      if checkbox.is_selected():
        checkbox.click()

class TestSystem(unittest.TestCase):

    def setUp(self):
        print("Setting up WebDriver")
        # Setup WebDriver once per test
        self.driver = webdriver.ChromiumEdge()
        print("WebDriver ready")

    def test_noPredictorModel(self):
        print("Running test_noPredictorModel")
        driver = self.driver

        driver.get("http://127.0.0.1:8080/")

        uncheck_all_checkboxes(driver)

        # Click the button with id "start-button"
        start_button = driver.find_element(By.ID, "start-button")
        start_button.click()
        print("Clicked start button")

        # Wait for the element with id "result" to be present
        result_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "results"))
        )

        print("Result element found")
        self.assertIsNotNone(result_element)

    def test_onePredictorModel(self):
        print("Running test_onePredictorModel")
        driver = self.driver

        driver.get("http://127.0.0.1:8080/")

        uncheck_all_checkboxes(driver)

        # Tick the checkbox with id "Brand-checkbox"
        brand_checkbox = driver.find_element(By.ID, "Brand-checkbox")
        brand_checkbox.click()
        print("Checked Brand-checkbox")

        # Click the button with id "start-button"
        start_button = driver.find_element(By.ID, "start-button")
        start_button.click()
        print("Clicked start button")

        # Wait for the element with id "result" to be present
        result_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "results"))
        )

        print("Result element found")
        self.assertIsNotNone(result_element)

    def test_multiPredictorModel(self):
        print("Running test_multiPredictorModel")
        driver = self.driver

        driver.get("http://127.0.0.1:8080/")

        uncheck_all_checkboxes(driver)

        # Tick the checkbox with id "Brand-checkbox"
        brand_checkbox = driver.find_element(By.ID, "Brand-checkbox")
        brand_checkbox.click()
        print("Checked Brand-checkbox")

        cylindersinEngine_checkbox = driver.find_element(By.ID, "CylindersinEngine-checkbox")
        cylindersinEngine_checkbox.click()
        print("Checked CylindersinEngine-checkbox")

        bodyType_checkbox = driver.find_element(By.ID, "BodyType-checkbox")
        bodyType_checkbox.click()
        print("Checked BodyType-checkbox")

        # Click the button with id "start-button"
        start_button = driver.find_element(By.ID, "start-button")
        start_button.click()
        print("Clicked start button")

        # Wait for the element with id "result" to be present
        result_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "results"))
        )

        print("Result element found")
        self.assertIsNotNone(result_element)

    def tearDown(self):
        print("Tearing down WebDriver")
        self.driver.quit()  # Ensure all browser windows are closed       

 
if __name__ in {"__main__", "__mp_main__"}:
  print("Running tests")
  
  guiInstance=gui.GUI()

  # Start the GUI first
  if __name__ in {"__main__"}:
    pass
  elif __name__ in {"__mp_main__"}:
    def run_loop():
      time.sleep(3) # Wait for the GUI to start

      unittest.main()  # Run the tests

      guiInstance.shutdown()  # Shutdown the GUI


    # Now start the loop in a separate thread
    test_thread = threading.Thread(target=run_loop)
    test_thread.start()
  else:
    pass 
    