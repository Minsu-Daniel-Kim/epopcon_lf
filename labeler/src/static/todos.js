var app = angular.module('app', []);

app.controller('demoController', function($scope, $http) {
    // initial items

    $scope.btnText = '틀림';

    $scope.items = [];

    $http.get("/items")
    .then(function(response) {

        $scope.items = response.data;


        console.log(response.data);

    });

//    $scope.selected = [];
    $scope.selected_dict = {};

    $scope.add = function(GOODS_NO, label) {




        $scope.selected_dict[GOODS_NO] = {
    	     'GOODS_NO': GOODS_NO,
    	     'STATUS': label
    	};

    	console.log(Object.keys($scope.selected_dict).length);

//        console.log(Object.keys($scope.selected_dict).length);

//        $scope.selected_dict[GOODS_NO] = {
//    	     'GOODS_NO': GOODS_NO,
//    	     'STATUS': label
//    	};

//        alert(GOODS_NO);
//
//    	$scope.selected.push({
//    	     'GOODS_NO': GOODS_NO,
//    	     'STATUS': label
//    	});
//        alert($scope.selected_dict);
//        alert(selected_dict);
    };

    $scope.done = function() {

       if ($scope.yourName == '' | $scope.yourName == undefined) {
            alert('실명을 적어주세요.');
       } else if (Object.keys($scope.selected_dict).length < 19) {
            alert('모두 선택해주세요.');
       } else {



        var bundle = [];

        for (var key in $scope.selected_dict) {
            bundle.push($scope.selected_dict[key])
        }



//        console.log("pushed to the server: " + {'username': $scope.yourName, 'bundle': bundle});
        $http.post("/save", {'username': $scope.yourName, 'bundle': bundle})
        .then(function(response) {
            console.log(response);
            $scope.items = [];
//            $scope.selected = [];
            $scope.selected_dict = {};
            $http.get("/items")
            .then(function(response) {
                $scope.items = response.data;
            });



        })
        }

    };
});