$(function(){
placeValuesOnSpots(data)
placeValuesOnSpotsClassification(data_classification)
console.log(data_classification)
console.log(data)
$('#expanded_container_id').hide();
$('#prof').html(data.linear.professional);


})



function showTable(){

$("#Regression_Outputs").toggle();
}
function hideTable(){

$('#Classification_Outputs').toggle();
}

function LinearExpanded()
{

  $('#prof').html=data.mars.expanded;
  $('#expanded_container_id').toggle();
  $('#expanded_info').html(data.linear.expanded);
  $("#expanded_image").attr("src",data.linear.fileName);
}
function LassoExpanded()
{
  $('#expanded_container_id').toggle();
  $('#expanded_info').html(data.lasso.expanded);
  $("#expanded_image").attr("src",data.lasso.fileName);
}
function MarsExpanded()
{
  $('#expanded_container_id').toggle();
  $('#expanded_info').html(data.mars.expanded);
  $("#expanded_image").attr("src",data.mars.fileName);
}
function SarimaxExpanded()
{
  $('#expanded_container_id').toggle();
  $('#expanded_info').html(data.sarimax.expanded);
  $("#expanded_image").attr("src",data.sarimax.fileName);
}
function DecisionExpanded()
{
  $('#expanded_container_id').toggle();
  $('#expanded_info').html(data_classification.decisiontree.expanded);
  $("#expanded_image").attr("src",data_classification.decisiontree.fileName);
}
function LogisticExpanded()
{
  $('#expanded_container_id').toggle();
  $('#expanded_info').html(data_classification.logistic.expanded);
  $("#expanded_image").attr("src",data_classification.logistic.fileName);
}
function RandomExpanded()
{
  $('#expanded_container_id').toggle();
  $('#expanded_info').html(data_classification.randomforest.expanded);
  $("#expanded_image").attr("src",data_classification.randomforest.fileName);
}

function placeValuesOnSpots(data){

$('#r2 > .d2').html(data.linear.accuracy)
$('#r2 > .d3').html(data.linear.MSE)
$('#r2 > .d4').html(data.linear.MAD)
$('#r2 > .d5').html(data.linear.MAPE)
$('#r3 > .d2').html(data.lasso.accuracy)
$('#r3 > .d3').html(data.lasso.MSE)
$('#r3 > .d4').html(data.lasso.MAD)
$('#r3 > .d5').html(data.lasso.MAPE)
$('#r4 > .d2').html(data.mars.accuracy)
$('#r4 > .d3').html(data.mars.MSE)
$('#r4 > .d4').html(data.mars.MAD)
$('#r4 > .d5').html(data.mars.MAPE)
$('#r5 > .d2').html(data.sarimax.accuracy)
$('#r5 > .d3').html(data.sarimax.MSE)
$('#r5 > .d4').html(data.sarimax.MAD)
$('#r5 > .d5').html(data.sarimax.MAPE)

}

function placeValuesOnSpotsClassification(data_classification){
$('#r6 > .d2').html(data_classification.randomforest.accuracy)
$('#r6 > .d3').html(data_classification.randomforest.TP)
$('#r6 > .d4').html(data_classification.randomforest.FP)
$('#r6 > .d5').html(data_classification.randomforest.TN)
$('#r6 > .d6').html(data_classification.randomforest.FN)

$('#r7 > .d2').html(data_classification.decisiontree.accuracy)
$('#r7 > .d3').html(data_classification.decisiontree.TP)
$('#r7 > .d4').html(data_classification.decisiontree.FP)
$('#r7 > .d5').html(data_classification.decisiontree.TN)
$('#r7 > .d6').html(data_classification.decisiontree.FN)

$('#r8 > .d2').html(data_classification.logistic.accuracy)
$('#r8 > .d3').html(data_classification.logistic.TP)
$('#r8 > .d4').html(data_classification.logistic.FP)
$('#r8 > .d5').html(data_classification.logistic.TN)
$('#r8 > .d6').html(data_classification.logistic.FN)
}


