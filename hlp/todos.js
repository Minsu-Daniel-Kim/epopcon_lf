var app = angular.module('app', []);

app.controller('demoController', function($scope, $http) {
    // initial items

    $scope.btnText = '틀림';

    $scope.items = [];

    $http.get("http://labler-e22fdc99-1.b358ec1f.cont.dockerapp.io:32769/items")
    .then(function(response) {

        $scope.items = response.data;
    });

    $scope.selected = [];

    $scope.add = function(GOODS_NO) {

    	$scope.selected.push(GOODS_NO);

    };

    $scope.done = function() {

        var bundle = [];
        console.log(bundle);
        for (var i = 0; i < $scope.items.length; i++) {

            for (var j = 0; j < $scope.selected.length; j++) {

                item = $scope.items[i];

                if (item.GOODS_NO == $scope.selected[j]) {

                    bundle.push({
                        'GOODS_NO': item.GOODS_NO,
                        'STATUS': '-1'
                    });

                } else {
                    bundle.push({
                        'GOODS_NO': item.GOODS_NO,
                        'STATUS': '1'
                    });
                }
            }
        }

        $http.post("http://labler-e22fdc99-1.b358ec1f.cont.dockerapp.io:32769/save", bundle)
        .then(function(response) {

            $scope.items = [];

            $http.get("http://labler-e22fdc99-1.b358ec1f.cont.dockerapp.io:32769/items")
            .then(function(response) {
                $scope.items = response.data;
            });



        })


    };
});