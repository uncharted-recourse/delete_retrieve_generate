// Initiate jQuery on load.
$(function() {
    // Style text with flask route
    $("#style").on("click", function(e) {
      e.preventDefault();
      var translateVal = document.getElementById("text-to-style").value;
      var quantity = document.getElementById("quantity").value;
      var k = document.getElementById("k").value;
      var temperature = document.getElementById("temperature").value;
      var styles = ['informal', 'formal', 'humorous', 'romantic'];
      var stylesChecked = []

      for (var i = 0; i < styles.length; i++) {
        var button = document.getElementById(styles[i]);
        if (button.checked == true) {
          stylesChecked.push(styles[i])
        }
      }
      // make sure at least one style checkbox has been checked
      try { (stylesChecked.length > 0);}
      catch(err) { document.getElementById("style-transfer-result").textContent = err.message;}
      
      var translateRequest = { 'input_text': translateVal, 'styles': stylesChecked, 'quantity': quantity, 
        'k': k, 'temperature': temperature};
      if (translateVal !== "") {
        $.ajax({
          url: '/style-transfer',
          method: 'POST',
          headers: {
              'Content-Type':'application/json'
          },
          dataType: 'json',
          data: JSON.stringify(translateRequest),
          success: function(data) {
            document.getElementById("style-transfer-result").textContent = data["output_texts"].join("\n");
          },
          error: function(err) {
            document.getElementById("style-transfer-result").textContent = err.message;
          }
        });
      };
    });
  })