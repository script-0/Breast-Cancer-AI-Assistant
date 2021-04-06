const sign_in_btn = document.querySelector("#sign-in-btn");
const sign_up_btn = document.querySelector("#sign-up-btn");
const container = document.querySelector(".container");
const mammogram = document.querySelector("#process_mammogram");
const biopsy = document.querySelector("#process_biopsy");
const upload = document.querySelector("#formFileSm");

const URL = 'http://127.0.0.1:5000/'

sign_up_btn.addEventListener("click", () => {
  container.classList.add("sign-up-mode");  
});

$('#process_biopsy').prop('disabled', true);

sign_in_btn.addEventListener("click", () => {
  container.classList.remove("sign-up-mode");
});

upload.addEventListener('change', function(e) {
  $('#process_biopsy').prop('disabled', true);
  const file = this.files[0];
  const  fileType = file['type'];
  const validImageTypes = ['image/gif', 'image/jpeg', 'image/png'];
  if (validImageTypes.includes(fileType)) {
    $('#process_biopsy').prop('disabled', false);
  }
},false);

biopsy.addEventListener('click', () => {
  $('#process_biopsy').prop('disabled', true);
  $('#process_biopsy').html('<span class="spinner-grow spinner-grow-sm" role="status" aria-hidden="true"></span> Loading...');
  var files=  document.querySelector("#formFileSm").files;
  var formData = new FormData();
  formData.append("file",files[0]);
  formData.append("test","test");
  var xhr = new XMLHttpRequest();
  xhr.open('post', URL+'biopsy', false);
  xhr.onload = function () {
    if (xhr.status !== 200) {
      // When XAuth service unavailable.
      console.log("Authentification service unavailable");
      
      return;
    }

    var reader  = new FileReader();
		
    reader.onloadend = function () {
      $('#normalImage')
        .prop('src',reader.result)  //set the scr prop.
        .prop('width', 216)  //set the width of the image
        .prop('height',200);  //set the height of the image
    }
    reader.readAsDataURL(files[0]);

    //
    console.log(this.responseText);

    $("#imageMask").prop('src',"data:image/png;base64,"+this.responseText)
                   .prop('width', 216)  //set the width of the image
                   .prop('height',200);  //set the height of the image;

    $('#biopsyModal').modal('hide');
    $('#biopsyresultModal').modal('show');

    $('#process_biopsy').html('Process');
    $('#process_biopsy').prop('disabled', false);
  }
  xhr.send(formData);
});

mammogram.addEventListener("click", () => {
  $('#process_mammogram').prop('disabled', true);
  $('#process_mammogram').html('<span class="spinner-grow spinner-grow-sm" role="status" aria-hidden="true"></span> Loading...');

  var params={
      age : parseFloat( $('#age').val() ),
      density : parseInt( $('#density').val() ),
      famhx : parseInt( $('#famhx').val() ),
      hrt : parseInt( $('#hrt').val() ), 
      prvmam: parseInt( $('#prvmam').val() ), 
      biophx: parseInt( $('#biophx').val() ),
      mammtype: parseInt( $('#mammtype').val() ), 
      bmi:parseFloat( $('#bmi').val() )
  }

  console.log(params)
  
  var xhr = new XMLHttpRequest();
  xhr.open('POST', URL+'mammogram', true);
  xhr.setRequestHeader('Content-type', 'application/json;charset=utf-8');
  xhr.onload = function () {
    if (xhr.status !== 200) {
      // When XAuth service unavailable.
      console.log("Authentification service unavailabe");
      
      return;
    }
    //
    var res = JSON.parse(this.responseText);
    console.log(res)
    console.log(res.assess_2)
    console.log(res.assess_3)
    console.log(res.assess_4)
    console.log(res.assess_5)
    response= [res]

    $('#class_1').html(
      '<div class="progress-bar" role="progressbar" style="width:'+Math.round(response[0].assess_1 * 100)+'%;" aria-valuenow="'+Math.round(response[0].assess_1 * 100)+'" aria-valuemin="0" aria-valuemax="100">'+Math.round(response[0].assess_1 * 100)+'%</div>'
    )
    $('#class_2').html(
      '<div class="progress-bar" role="progressbar" style="width:'+Math.round(response[0].assess_2 * 100)+'%;" aria-valuenow="'+Math.round(response[0].assess_2 * 100)+'" aria-valuemin="0" aria-valuemax="100">'+Math.round(response[0].assess_2 * 100)+'%</div>'
    )
    $('#class_3').html(
      '<div class="progress-bar" role="progressbar" style="width:'+Math.round(response[0].assess_3 * 100)+'%;" aria-valuenow="'+Math.round(response[0].assess_3* 100)+'" aria-valuemin="0" aria-valuemax="100">'+Math.round(response[0].assess_3 * 100)+'%</div>'
    )
    $('#class_4').html(
      '<div class="progress-bar" role="progressbar" style="width:'+Math.round(response[0].assess_4 * 100)+'%;" aria-valuenow="'+Math.round(response[0].assess_4* 100)+'" aria-valuemin="0" aria-valuemax="100">'+Math.round(response[0].assess_4 * 100)+'%</div>'
    )
    $('#class_5').html(
      '<div class="progress-bar" role="progressbar" style="width:'+Math.round(response[0].assess_5 * 100)+'%;" aria-valuenow="'+Math.round(response[0].assess_5 * 100)+'" aria-valuemin="0" aria-valuemax="100">'+Math.round(response[0].assess_5 * 100)+'%</div>'
    )
    $('#class_6').html(
      '<div class="progress-bar" role="progressbar" style="width:'+Math.round(response[0].assess_6 * 100)+'%;" aria-valuenow="'+Math.round(response[0].assess_6 * 100)+'" aria-valuemin="0" aria-valuemax="100">'+Math.round(response[0].assess_6 * 100)+'%</div>'
    )

    $('#exampleModal').modal('hide');
    $('#resultModal').modal('show');
    $('#process_mammogram').html('Process');
    $('#process_mammogram').prop('disabled', false);
  }
  xhr.send( JSON.stringify(params) );
});
